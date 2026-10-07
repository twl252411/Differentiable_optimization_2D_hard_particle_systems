r"""Generate image/random RVE data and CAE files with size-tagged filenames.

By default, outputs are written under ``outputs/nonlinear_damage_r5_l100``
relative to the repository root. Use ``--out-dir`` to choose another location.

Different RVE sizes are separated by filename tags, for example:

    image_L080_r5_vf050_N041_rve_001.npz
    random_L100_r5_vf050_N064_rve_001.npz

This avoids reusing or overwriting old npz files when rve_size changes.

Normal Python stage:

    python workflows\generate_rve.py --rve-size 80 --force

Abaqus CAE stage:

    cd outputs\nonlinear_damage_r5_l100\cae
    abaqus cae noGUI=generate_periodic_disk_rves_only.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import deep_copy_config, load_config  # noqa: E402
from geometry import mic_delta_np, wrap_points  # noqa: E402
from optimizer import optimize_rve_2d, save_result  # noqa: E402


DEFAULT_CONFIG = load_config(ROOT / "configs" / "default_2d.yaml")
DETECTED_CONFIG = load_config(ROOT / "configs" / "from_detected_fibers.yaml")


def default_config() -> dict[str, Any]:
    return deep_copy_config(DEFAULT_CONFIG)


def detected_config() -> dict[str, Any]:
    return deep_copy_config(DETECTED_CONFIG)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

OUT_ROOT = ROOT / "outputs" / "nonlinear_damage_r5_l100"

RADIUS_UM = 5.0
HARD_CORE_RADIUS_SCALE = 1.035
AREA_FRACTION_TARGET = 0.50

SAMPLES_PER_CASE = 1
CASES = ("image", "random")

CENTER_DISTANCE_TOL = 1.0e-6
MAX_RETRY_PER_SAMPLE = 5

# These are updated by set_problem_parameters().
LENGTH_UM = np.sqrt(RADIUS_UM**2 * np.pi*30/AREA_FRACTION_TARGET)
NUM_PARTICLES = 0
AREA_FRACTION_REALIZED = 0.0
MIN_CENTER_DISTANCE_UM = 0.0
RVE_TAG = ""


def set_problem_parameters(length_um: float, area_fraction: float, num_particles: int | None = None) -> None:
    """Update global RVE parameters and the filename tag."""

    global LENGTH_UM
    global AREA_FRACTION_TARGET
    global NUM_PARTICLES
    global AREA_FRACTION_REALIZED
    global MIN_CENTER_DISTANCE_UM
    global RVE_TAG

    LENGTH_UM = float(length_um)
    AREA_FRACTION_TARGET = float(area_fraction)

    if num_particles is None:
        NUM_PARTICLES = int(round(AREA_FRACTION_TARGET * LENGTH_UM * LENGTH_UM / (math.pi * RADIUS_UM * RADIUS_UM)))
    else:
        NUM_PARTICLES = int(num_particles)
        if NUM_PARTICLES <= 1:
            raise ValueError("--num-particles must be greater than one.")
    AREA_FRACTION_REALIZED = NUM_PARTICLES * math.pi * RADIUS_UM * RADIUS_UM / (LENGTH_UM * LENGTH_UM)
    AREA_FRACTION_TARGET = AREA_FRACTION_REALIZED
    MIN_CENTER_DISTANCE_UM = 2.0 * RADIUS_UM * HARD_CORE_RADIUS_SCALE

    length_tag = "L%03d" % int(round(LENGTH_UM))
    radius_tag = "r%g" % RADIUS_UM
    radius_tag = radius_tag.replace(".", "p")
    vf_tag = "vf%03d" % int(round(100.0 * AREA_FRACTION_TARGET))
    n_tag = "N%03d" % int(NUM_PARTICLES)
    RVE_TAG = "%s_%s_%s_%s" % (length_tag, radius_tag, vf_tag, n_tag)


def to_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def safe_relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return to_posix(path)


def scale_config_lengths(cfg: dict[str, Any], factor: float) -> None:
    """Scale all length-dependent descriptor parameters after changing box size."""

    for key in ("rdf", "anisotropic_rdf"):
        for item in ("r_min", "r_max", "sigma", "sigma_r"):
            if item in cfg.get(key, {}) and cfg[key].get(item) is not None:
                cfg[key][item] = float(cfg[key][item]) * factor

    for item in ("tau", "grid_min", "grid_max", "bandwidth"):
        if item in cfg.get("nearest_neighbor", {}) and cfg["nearest_neighbor"].get(item) is not None:
            cfg["nearest_neighbor"][item] = float(cfg["nearest_neighbor"][item]) * factor

    for item in ("cutoff_radius", "bandwidth"):
        if item in cfg.get("coordination", {}) and cfg["coordination"].get(item) is not None:
            cfg["coordination"][item] = float(cfg["coordination"][item]) * factor

    area_factor = factor * factor
    for item in ("grid_min", "grid_max", "bandwidth"):
        if item in cfg.get("voronoi_area_distribution", {}) and cfg["voronoi_area_distribution"].get(item) is not None:
            cfg["voronoi_area_distribution"][item] = float(cfg["voronoi_area_distribution"][item]) * area_factor


def least_crowded_subset(points: np.ndarray, box_size: float, count: int) -> np.ndarray:
    """Remove locally crowded detected points first and keep a deterministic subset."""

    keep = list(range(points.shape[0]))
    box = np.asarray([box_size, box_size], dtype=float)

    while len(keep) > count:
        pts = points[keep]
        delta = mic_delta_np(pts[:, None, :] - pts[None, :, :], box)
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)
        nearest = dist.min(axis=1)
        remove_local = int(np.argmin(nearest))
        del keep[remove_local]

    return points[np.asarray(keep, dtype=int)]


def write_image_target_subset(out_dir: Path) -> Path:
    """Write the image-derived statistical reference point file scaled to the current RVE size.

    The detected image points are used only to define the target statistics.
    The optimized/generated RVE may have a different particle count.
    """

    base_cfg = detected_config()
    base_box = float(base_cfg["geometry"]["box_size"])
    source = ROOT / base_cfg["target"]["points_path"]

    pts = np.loadtxt(source, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    pts = wrap_points(pts[:, :2], base_box)

    if pts.shape[0] < NUM_PARTICLES:
        print(
            "[warn] image reference has only %d points, less than requested NUM_PARTICLES=%d; "
            "using all detected image points only as the statistical reference, while the optimized RVE may use a different particle count."
            % (pts.shape[0], NUM_PARTICLES),
            flush=True,
        )
        subset = pts
    else:
        subset = least_crowded_subset(pts, base_box, NUM_PARTICLES)
    scaled = wrap_points(subset * (LENGTH_UM / base_box), LENGTH_UM)

    target_dir = out_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / ("image_target_points_%s.txt" % RVE_TAG)
    np.savetxt(out, scaled, fmt="%.10f")

    return out.resolve()


def image_config(out_dir: Path, samples: int) -> dict[str, Any]:
    """Configuration for image-derived target RVE generation."""

    cfg = detected_config()
    base_box = float(cfg["geometry"]["box_size"])
    scale_config_lengths(cfg, LENGTH_UM / base_box)

    target_path = write_image_target_subset(out_dir)
    cache_dir = out_dir / "_optimizer_cache" / RVE_TAG / "image"

    cfg["project"] = {
        "output_dir": safe_relative_or_absolute(cache_dir, ROOT),
        "num_samples": int(samples),
        "seed": 81001,
    }
    cfg["geometry"] = {
        "box_size": LENGTH_UM,
        "diameter": None,
        "radius": RADIUS_UM,
        "statistics_radius_scale": HARD_CORE_RADIUS_SCALE,
        "area_fraction": None,
        "num_particles": NUM_PARTICLES,
    }
    cfg["target"] = {
        "source": "points_file",
        "points_path": to_posix(target_path),
    }

    cfg["phase1"]["max_iters"] = max(int(cfg["phase1"].get("max_iters", 2400)), 3600)
    cfg["phase2"]["terminal_projection_iters"] = max(
        int(cfg["phase2"].get("terminal_projection_iters", 1200)), 1800
    )

    return cfg


def random_config(out_dir: Path, samples: int) -> dict[str, Any]:
    """Configuration for random-reference target RVE generation."""

    cfg = default_config()
    base_box = float(cfg["geometry"]["box_size"])
    scale_config_lengths(cfg, LENGTH_UM / base_box)

    cache_dir = out_dir / "_optimizer_cache" / RVE_TAG / "random"

    cfg["project"] = {
        "output_dir": safe_relative_or_absolute(cache_dir, ROOT),
        "num_samples": int(samples),
        "seed": 82001,
    }
    cfg["geometry"] = {
        "box_size": LENGTH_UM,
        "diameter": None,
        "radius": RADIUS_UM,
        "statistics_radius_scale": HARD_CORE_RADIUS_SCALE,
        "area_fraction": None,
        "num_particles": NUM_PARTICLES,
    }
    cfg["target"] = {"source": "random", "seed": 92001}

    cfg["phase1"]["max_iters"] = max(int(cfg["phase1"].get("max_iters", 1600)), 3600)
    cfg["phase2"]["terminal_projection_iters"] = max(
        int(cfg["phase2"].get("terminal_projection_iters", 800)), 1800
    )

    return cfg


def named_npz_path(out_dir: Path, case: str, index: int) -> Path:
    return out_dir / "npz" / ("%s_%s_rve_%03d.npz" % (case, RVE_TAG, index + 1))


def named_summary_path(out_dir: Path, case: str, index: int) -> Path:
    return out_dir / "npz" / ("%s_%s_rve_%03d_summary.json" % (case, RVE_TAG, index + 1))


def named_distance_report_path(out_dir: Path, case: str, index: int) -> Path:
    return out_dir / "npz" / ("%s_%s_rve_%03d_closest_center_pairs.csv" % (case, RVE_TAG, index + 1))


def cache_case_dir(out_dir: Path, case: str) -> Path:
    return out_dir / "_optimizer_cache" / RVE_TAG / case


def cache_npz_path(out_dir: Path, case: str, index: int) -> Path:
    return cache_case_dir(out_dir, case) / ("rve_%04d.npz" % index)


def cache_summary_path(out_dir: Path, case: str, index: int) -> Path:
    return cache_case_dir(out_dir, case) / ("rve_%04d_summary.json" % index)


def minimum_periodic_center_distance(points: np.ndarray, box_size: float) -> tuple[float, int, int]:
    """Return the minimum MIC center distance and the corresponding particle pair."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise RuntimeError("points must have shape (N, 2).")
    if pts.shape[0] < 2:
        return float("inf"), -1, -1

    box = np.asarray([box_size, box_size], dtype=float)
    delta = mic_delta_np(pts[:, None, :2] - pts[None, :, :2], box)
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, np.inf)
    k = int(np.argmin(dist))
    i, j = divmod(k, dist.shape[1])
    return float(dist[i, j]), int(i), int(j)


def write_center_distance_report(path: Path, points: np.ndarray, box_size: float, limit: int = 20) -> None:
    """Write the closest MIC center pairs for debugging."""

    pts = np.asarray(points, dtype=float)
    box = np.asarray([box_size, box_size], dtype=float)
    delta = mic_delta_np(pts[:, None, :2] - pts[None, :, :2], box)
    dist = np.linalg.norm(delta, axis=-1)

    rows = []
    n = pts.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            rows.append((float(dist[i, j]), i, j, pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1]))
    rows.sort(key=lambda row: row[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "distance_mic", "i", "j", "xi", "yi", "xj", "yj", "min_allowed"])
        for rank, row in enumerate(rows[:limit], 1):
            writer.writerow([rank] + list(row) + [MIN_CENTER_DISTANCE_UM])


def load_points_from_npz(npz_path: Path) -> tuple[np.ndarray, float]:
    data = np.load(npz_path)
    points = np.asarray(data["points_final"], dtype=float)
    box = float(np.asarray(data["box_size"]).reshape(-1)[0])
    return points, box


def validate_npz_centers(npz_path: Path, report_path: Path) -> tuple[bool, float, int, int]:
    """Validate npz centers after scaling to the final target LENGTH_UM."""

    points, box = load_points_from_npz(npz_path)
    scaled = wrap_points(points * (LENGTH_UM / box), LENGTH_UM)
    dmin, i, j = minimum_periodic_center_distance(scaled, LENGTH_UM)

    if dmin < MIN_CENTER_DISTANCE_UM - CENTER_DISTANCE_TOL:
        write_center_distance_report(report_path, scaled, LENGTH_UM)
        return False, dmin, i, j

    return True, dmin, i, j


def copy_cache_result_to_named_files(out_dir: Path, case: str, index: int) -> tuple[Path, Path]:
    """Copy optimizer output rve_0000.* to size-tagged filenames."""

    src_npz = cache_npz_path(out_dir, case, index)
    src_summary = cache_summary_path(out_dir, case, index)
    dst_npz = named_npz_path(out_dir, case, index)
    dst_summary = named_summary_path(out_dir, case, index)

    if not src_npz.is_file():
        raise RuntimeError("Missing optimizer npz: %s" % src_npz)
    if not src_summary.is_file():
        raise RuntimeError("Missing optimizer summary: %s" % src_summary)

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_npz, dst_npz)
    shutil.copy2(src_summary, dst_summary)

    return dst_npz, dst_summary


def prepare_rve_samples(out_dir: Path, samples: int, force: bool) -> list[dict[str, Any]]:
    """Generate or reuse size-tagged image/random RVE npz files."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)

    configs = {
        "image": image_config(out_dir, samples),
        "random": random_config(out_dir, samples),
    }

    manifest_rows: list[dict[str, Any]] = []

    for case in CASES:
        cfg = configs[case]
        case_cache = cache_case_dir(out_dir, case)
        case_cache.mkdir(parents=True, exist_ok=True)
        (case_cache / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        for index in range(samples):
            npz_path = named_npz_path(out_dir, case, index)
            summary_path = named_summary_path(out_dir, case, index)
            distance_report = named_distance_report_path(out_dir, case, index)

            status = ""
            dmin = float("nan")
            pair_i = -1
            pair_j = -1

            use_existing = npz_path.exists() and summary_path.exists() and not force
            if use_existing:
                ok, dmin, pair_i, pair_j = validate_npz_centers(npz_path, distance_report)
                if ok:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    status = "existing_valid"
                else:
                    print(
                        "[warn] existing %s sample %d violates hard-core after scaling: "
                        "dmin=%.12g < %.12g, pair=(%d,%d). Regenerating. Report: %s"
                        % (case, index + 1, dmin, MIN_CENTER_DISTANCE_UM, pair_i, pair_j, distance_report),
                        flush=True,
                    )
                    use_existing = False

            if not use_existing:
                summary = None
                last_error = None
                for attempt in range(MAX_RETRY_PER_SAMPLE):
                    seed_offset = index + 10000 * attempt
                    try:
                        result = optimize_rve_2d(copy.deepcopy(cfg), seed_offset=seed_offset)
                        save_result(result, case_cache, index=index)
                        copy_cache_result_to_named_files(out_dir, case, index)

                        ok, dmin, pair_i, pair_j = validate_npz_centers(npz_path, distance_report)
                        if ok:
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                            status = "created" if attempt == 0 else "created_retry_%d" % attempt
                            break

                        last_error = (
                            "dmin=%.12g < %.12g, pair=(%d,%d), report=%s"
                            % (dmin, MIN_CENTER_DISTANCE_UM, pair_i, pair_j, distance_report)
                        )
                        print("[warn] invalid generated %s sample %d attempt %d: %s" % (case, index + 1, attempt, last_error), flush=True)
                    except Exception as exc:
                        last_error = repr(exc)
                        print("[warn] failed generated %s sample %d attempt %d: %s" % (case, index + 1, attempt, last_error), flush=True)

                if summary is None:
                    raise RuntimeError(
                        "Failed to generate valid %s sample %d after %d attempts. Last error: %s"
                        % (case, index + 1, MAX_RETRY_PER_SAMPLE, last_error)
                    )

            row = {
                "case": case,
                "rve_id": index + 1,
                "status": status,
                "npz": safe_relative_or_absolute(npz_path, ROOT),
                "summary": safe_relative_or_absolute(summary_path, ROOT),
                "rve_tag": RVE_TAG,
                "rve_size_um": LENGTH_UM,
                "num_particles": int(summary["num_particles"]),
                "area_fraction": float(summary["area_fraction"]),
                "loss_overlap": float(summary["final_terms"]["loss_overlap"]),
                "min_center_distance": float(dmin),
                "min_allowed_center_distance": float(MIN_CENTER_DISTANCE_UM),
                "closest_pair_i": int(pair_i),
                "closest_pair_j": int(pair_j),
            }
            manifest_rows.append(row)
            print(json.dumps(row), flush=True)

    manifest_path = out_dir / ("rve_manifest_%s.json" % RVE_TAG)
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    return manifest_rows


def write_scaled_centers_for_abaqus(out_dir: Path, samples: int) -> list[dict[str, Any]]:
    """Write Abaqus manifest entries without duplicating center coordinates on disk."""

    rows: list[dict[str, Any]] = []
    cae_dir = out_dir / "cae"
    cae_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        for index in range(samples):
            npz_path = named_npz_path(out_dir, case, index)
            data = np.load(npz_path)

            points = np.asarray(data["points_final"], dtype=float)
            box = float(np.asarray(data["box_size"]).reshape(-1)[0])
            hard_core_scale = float(np.asarray(data["radius_scale"])) if "radius_scale" in data.files else HARD_CORE_RADIUS_SCALE

            scaled = wrap_points(points * (LENGTH_UM / box), LENGTH_UM)
            dmin_scaled, pair_i, pair_j = minimum_periodic_center_distance(scaled, LENGTH_UM)

            centers_report = cae_dir / ("%s_%s_rve_%03d_closest_center_pairs.csv" % (case, RVE_TAG, index + 1))
            if dmin_scaled < MIN_CENTER_DISTANCE_UM - CENTER_DISTANCE_TOL:
                write_center_distance_report(centers_report, scaled, LENGTH_UM)
                raise RuntimeError(
                    "Scaled centers violate hard-core for %s sample %d: "
                    "dmin=%.12g < %.12g, pair=(%d,%d). Report: %s"
                    % (case, index + 1, dmin_scaled, MIN_CENTER_DISTANCE_UM, pair_i, pair_j, centers_report)
                )

            job_index = len(rows) + 1

            cae_name = "%s_%s_rve_%03d.cae" % (case, RVE_TAG, index + 1)

            meta = {
                "job_index": job_index,
                "case": case,
                "rve_id": index + 1,
                "rve_tag": RVE_TAG,
                "rve_size_um": LENGTH_UM,
                "source_npz": safe_relative_or_absolute(npz_path, ROOT),
                "cae_name": cae_name,
                "num_particles": int(points.shape[0]),
                "box_scaled_um": LENGTH_UM,
                "radius_scaled_um": RADIUS_UM,
                "hard_core_radius_scale": hard_core_scale,
                "hard_core_radius_scaled_um": RADIUS_UM * hard_core_scale,
                "area_fraction": AREA_FRACTION_REALIZED,
                "min_center_distance_scaled_um": float(dmin_scaled),
                "min_allowed_center_distance_um": float(MIN_CENTER_DISTANCE_UM),
                "closest_pair_i": int(pair_i),
                "closest_pair_j": int(pair_j),
            }
            (cae_dir / ("%s_%s_rve_%03d_meta.json" % (case, RVE_TAG, index + 1))).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rows.append(meta)

    manifest_path = cae_dir / ("case_manifest_%s.json" % RVE_TAG)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # The Abaqus noGUI script always reads case_manifest.json.  It is overwritten
    # by the currently selected RVE size, while the size-tagged manifest is kept.
    (cae_dir / "case_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


GENERATE_RVES_ONLY = r"""
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import csv
import json
import os
import numpy as np

executeOnCaeStartup()

R_LENGTH = __R_LENGTH__
RADIUS = __RADIUS__
HARD_CORE_RADIUS_SCALE = __HARD_CORE_RADIUS_SCALE__


def wrap_coord(value, length):
    while value < 0.0:
        value += length
    while value >= length:
        value -= length
    return value


def read_centers_from_npz(path):
    data = np.load(path)
    points = np.asarray(data["points_final"], dtype=float)
    box = float(np.asarray(data["box_size"]).reshape(-1)[0])
    scale = R_LENGTH / box
    rows = []
    for point in points:
        rows.append((
            wrap_coord(float(point[0]) * scale, R_LENGTH),
            wrap_coord(float(point[1]) * scale, R_LENGTH),
        ))
    return rows


def minimum_periodic_distance(centers, length):
    dmin = 1.0e99
    pair = (-1, -1)
    for i in range(len(centers) - 1):
        xi, yi = centers[i]
        for j in range(i + 1, len(centers)):
            xj, yj = centers[j]
            dx = xi - xj
            dy = yi - yj
            dx -= length * round(dx / length)
            dy -= length * round(dy / length)
            d = (dx * dx + dy * dy) ** 0.5
            if d < dmin:
                dmin = d
                pair = (i, j)
    return dmin, pair


def write_center_distance_report(path, centers, length):
    rows = []
    for i in range(len(centers) - 1):
        xi, yi = centers[i]
        for j in range(i + 1, len(centers)):
            xj, yj = centers[j]
            dx = xi - xj
            dy = yi - yj
            dx -= length * round(dx / length)
            dy -= length * round(dy / length)
            d = (dx * dx + dy * dy) ** 0.5
            rows.append((d, i, j, xi, yi, xj, yj))
    rows.sort(key=lambda row: row[0])
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "distance_mic", "i", "j", "xi", "yi", "xj", "yj", "min_allowed"])
        for rank, row in enumerate(rows[:20], 1):
            writer.writerow([rank] + list(row) + [2.0 * RADIUS * HARD_CORE_RADIUS_SCALE])


def periodic_images(centers, radius, length):
    out = []
    seen = set()
    for x, y in centers:
        for sx in (-length, 0.0, length):
            for sy in (-length, 0.0, length):
                xx = x + sx
                yy = y + sy
                if xx >= -radius and xx <= length + radius and yy >= -radius and yy <= length + radius:
                    key = (round(xx, 8), round(yy, 8))
                    if key not in seen:
                        seen.add(key)
                        out.append((xx, yy))
    return out


def make_circle_part(model, radius):
    sketch = model.ConstrainedSketch(name="__circle__", sheetSize=2.5 * R_LENGTH)
    sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(radius, 0.0))
    part = model.Part(name="CircleSeed", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches["__circle__"]
    return part


def make_rectangle_part(model, name, length):
    sketch = model.ConstrainedSketch(name="__rect__", sheetSize=2.5 * R_LENGTH)
    sketch.rectangle(point1=(0.0, 0.0), point2=(length, length))
    part = model.Part(name=name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches["__rect__"]
    return part


def generate_one(meta):
    index = int(meta["job_index"])
    case = str(meta["case"])
    rve_id = int(meta["rve_id"])
    source_npz = str(meta["source_npz"])
    cae_name = str(meta["cae_name"])

    Mdb()
    model = mdb.models["Model-1"]

    centers = read_centers_from_npz(source_npz)
    min_allowed = 2.0 * RADIUS * HARD_CORE_RADIUS_SCALE
    dmin, pair = minimum_periodic_distance(centers, R_LENGTH)
    if dmin < min_allowed - 1.0e-6:
        report = cae_name.replace(".cae", "_abaqus_center_distance_error.csv")
        write_center_distance_report(report, centers, R_LENGTH)
        raise RuntimeError(
            "Input centers violate hard-core before CAE generation: "
            "case=%s rve_id=%d dmin=%.12g min_allowed=%.12g pair=%s report=%s"
            % (case, rve_id, dmin, min_allowed, str(pair), report)
        )

    images = periodic_images(centers, RADIUS, R_LENGTH)

    circle = make_circle_part(model, RADIUS)
    assembly = model.rootAssembly

    instances = ()
    for i, (x, y) in enumerate(images):
        name = "Disk-%d" % (i + 1)
        assembly.Instance(name=name, part=circle, dependent=OFF)
        assembly.translate(instanceList=(name,), vector=(x, y, 0.0))
        instances = instances + (assembly.instances[name],)

    assembly.InstanceFromBooleanMerge(
        name="FiberAll",
        instances=instances,
        originalInstances=DELETE,
        domain=GEOMETRY,
    )

    if "FiberAll-1" in assembly.instances.keys():
        del assembly.instances["FiberAll-1"]
    if "CircleSeed" in model.parts.keys():
        del model.parts["CircleSeed"]

    make_rectangle_part(model, "Domain", R_LENGTH)
    assembly.Instance(name="Domain-1", part=model.parts["Domain"], dependent=OFF)
    assembly.Instance(name="FiberAll-clip-1", part=model.parts["FiberAll"], dependent=OFF)

    assembly.InstanceFromBooleanCut(
        name="FiberOutside",
        instanceToBeCut=assembly.instances["FiberAll-clip-1"],
        cuttingInstances=(assembly.instances["Domain-1"],),
        originalInstances=DELETE,
    )

    assembly.Instance(name="FiberAll-main-1", part=model.parts["FiberAll"], dependent=OFF)

    assembly.InstanceFromBooleanCut(
        name="Part-2",
        instanceToBeCut=assembly.instances["FiberAll-main-1"],
        cuttingInstances=(assembly.instances["FiberOutside-1"],),
        originalInstances=DELETE,
    )

    for pname in ("FiberAll", "Domain", "FiberOutside"):
        if pname in model.parts.keys():
            del model.parts[pname]

    if "Part-2-1" in assembly.instances.keys():
        assembly.deleteFeatures(("Part-2-1",))

    make_rectangle_part(model, "MatrixDomain", R_LENGTH)
    assembly.Instance(name="MatrixDomain-1", part=model.parts["MatrixDomain"], dependent=OFF)
    assembly.Instance(name="Part-2-1", part=model.parts["Part-2"], dependent=OFF)

    assembly.InstanceFromBooleanCut(
        name="Part-1",
        instanceToBeCut=assembly.instances["MatrixDomain-1"],
        cuttingInstances=(assembly.instances["Part-2-1"],),
        originalInstances=DELETE,
    )

    if "MatrixDomain" in model.parts.keys():
        del model.parts["MatrixDomain"]

    if "Part-1-1" in assembly.instances.keys():
        assembly.deleteFeatures(("Part-1-1",))

    assembly.Instance(name="Part-1-1", part=model.parts["Part-1"], dependent=OFF)
    assembly.Instance(name="Part-2-1", part=model.parts["Part-2"], dependent=OFF)

    area_f = model.parts["Part-2"].getMassProperties()["area"]
    area_m = model.parts["Part-1"].getMassProperties()["area"]
    vf = area_f / (area_f + area_m)

    print("RVE generated: index=%d case=%s rve_id=%d centers=%d images=%d radius=%g vf=%g -> %s" % (
        index, case, rve_id, len(centers), len(images), RADIUS, vf, cae_name
    ))

    mdb.saveAs(pathName=cae_name)


with open("case_manifest.json", "r") as fh:
    manifest = json.load(fh)

for meta in manifest:
    cae_name = str(meta["cae_name"])
    if os.path.exists(cae_name):
        print("Skipping existing %s" % cae_name)
        continue
    generate_one(meta)

print("RVE-only CAE generation finished.")
"""


def write_abaqus_rve_script(out_dir: Path) -> Path:
    cae_dir = out_dir / "cae"
    if not (cae_dir / "case_manifest.json").is_file():
        raise RuntimeError("Missing %s" % (cae_dir / "case_manifest.json"))

    text = GENERATE_RVES_ONLY
    text = text.replace("__R_LENGTH__", "%.12g" % LENGTH_UM)
    text = text.replace("__RADIUS__", "%.12g" % RADIUS_UM)
    text = text.replace("__HARD_CORE_RADIUS_SCALE__", "%.12g" % HARD_CORE_RADIUS_SCALE)

    script_path = cae_dir / "generate_periodic_disk_rves_only.py"
    script_path.write_text(text.lstrip(), encoding="utf-8")

    bat_text = """@echo off
cd /d "%~dp0"
abaqus cae noGUI=generate_periodic_disk_rves_only.py
pause
"""
    bat_path = cae_dir / "generate_rves_only.bat"
    bat_path.write_text(bat_text, encoding="ascii")

    ps_text = """$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$abaqusCommand = if ($env:ABAQUS_COMMAND) { $env:ABAQUS_COMMAND } else { 'abaqus' }
& $abaqusCommand cae noGUI=generate_periodic_disk_rves_only.py
"""
    ps_path = cae_dir / "generate_rves_only.ps1"
    ps_path.write_text(ps_text, encoding="utf-8")

    return script_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--rve-size", type=float, default=LENGTH_UM)
    parser.add_argument("--area-fraction", type=float, default=AREA_FRACTION_TARGET)
    parser.add_argument(
        "--num-particles",
        type=int,
        default=None,
        help="Override the particle count explicitly. When provided, the realized area fraction is recomputed from N, r, and L.",
    )
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CASE)
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Only rewrite center CSV and Abaqus generation script from existing size-tagged npz files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate RVE npz files even if existing size-tagged files are present.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if args.rve_size <= 0:
        raise ValueError("--rve-size must be positive.")
    if args.area_fraction <= 0.0 or args.area_fraction >= 0.90:
        raise ValueError("--area-fraction should be in (0, 0.90).")
    if args.num_particles is not None and args.num_particles <= 1:
        raise ValueError("--num-particles must be greater than one.")

    set_problem_parameters(args.rve_size, args.area_fraction, num_particles=args.num_particles)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_optimize:
        prepare_rve_samples(out_dir, samples=args.samples, force=args.force)

    rows = write_scaled_centers_for_abaqus(out_dir, samples=args.samples)
    script_path = write_abaqus_rve_script(out_dir)

    info = {
        "out_dir": to_posix(out_dir),
        "cases": list(CASES),
        "samples_per_case": int(args.samples),
        "total_rves": len(rows),
        "rve_tag": RVE_TAG,
        "num_particles": NUM_PARTICLES,
        "radius_um": RADIUS_UM,
        "rve_size_um": LENGTH_UM,
        "area_fraction_target": AREA_FRACTION_TARGET,
        "area_fraction_realized": AREA_FRACTION_REALIZED,
        "hard_core_radius_scale": HARD_CORE_RADIUS_SCALE,
        "min_allowed_center_distance_um": MIN_CENTER_DISTANCE_UM,
        "npz_dir": to_posix(out_dir / "npz"),
        "cae_dir": to_posix(out_dir / "cae"),
        "abaqus_script": to_posix(script_path),
    }
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
