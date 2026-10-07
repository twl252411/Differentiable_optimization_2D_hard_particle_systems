"""Report optimizer-consistent and classical particle statistics for 2D RVEs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from config import load_config
from optimizer import normalize_config
from descriptor_statistics import (
    build_targets_from_reference,
    overlap_loss,
    smooth_coordination_numbers,
    soft_nearest_neighbor_distances,
    soft_voronoi_areas,
)


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _centers_to_edges(grid: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    grid = _as_float_array(grid)
    if grid.size == 1:
        width = 1.0
        edges = np.asarray([grid[0] - 0.5 * width, grid[0] + 0.5 * width], dtype=np.float64)
    else:
        mids = 0.5 * (grid[:-1] + grid[1:])
        first = grid[0] - (mids[0] - grid[0])
        last = grid[-1] + (grid[-1] - mids[-1])
        edges = np.concatenate([[first], mids, [last]]).astype(np.float64)
    if lower is not None:
        edges[0] = float(lower)
    if upper is not None:
        edges[-1] = float(upper)
    return edges


def _upper_pair_vectors_np(points: np.ndarray, box_size: float, periodic: bool = True) -> np.ndarray:
    n = len(points)
    i, j = np.triu_indices(n, k=1)
    delta = points[j] - points[i]
    if periodic:
        delta = delta - float(box_size) * np.round(delta / float(box_size))
    return delta


def _nearest_distances_np(points: np.ndarray, box_size: float, periodic: bool = True) -> np.ndarray:
    values = np.empty((len(points),), dtype=np.float64)
    for i in range(len(points)):
        delta = points - points[i]
        if periodic:
            delta = delta - float(box_size) * np.round(delta / float(box_size))
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        dist[i] = np.inf
        values[i] = float(np.min(dist))
    return values


def exact_hard_core_summary(
    points: np.ndarray,
    radii: np.ndarray | float,
    box_size: float,
    *,
    periodic: bool = True,
) -> dict[str, Any]:
    points = _as_float_array(points)
    if np.isscalar(radii):
        radii_arr = np.full((len(points),), float(radii), dtype=np.float64)
    else:
        radii_arr = _as_float_array(radii)

    violating_pairs = 0
    max_overlap = 0.0
    min_gap = np.inf
    min_center_distance = np.inf
    min_distance_pair: list[int] | None = None

    for i in range(len(points) - 1):
        delta = points[i + 1 :] - points[i]
        if periodic:
            delta = delta - float(box_size) * np.round(delta / float(box_size))
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        cutoff = radii_arr[i] + radii_arr[i + 1 :]
        gap = dist - cutoff
        if dist.size:
            local = int(np.argmin(dist))
            if float(dist[local]) < min_center_distance:
                min_center_distance = float(dist[local])
                min_distance_pair = [int(i), int(i + 1 + local)]
            min_gap = min(min_gap, float(np.min(gap)))
            overlap = -gap
            max_overlap = max(max_overlap, float(np.max(overlap)))
            violating_pairs += int(np.sum(gap < -1.0e-10))

    return {
        "periodic": bool(periodic),
        "violating_pairs": int(violating_pairs),
        "max_overlap": float(max(0.0, max_overlap)),
        "min_gap": float(min_gap),
        "min_center_distance": float(min_center_distance),
        "min_distance_pair": min_distance_pair,
    }


def original_rdf(points: np.ndarray, box_size: float, r_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _upper_pair_vectors_np(points, box_size, periodic=True)
    distances = np.sqrt(np.sum(vectors * vectors, axis=1))
    edges = _centers_to_edges(r_grid, lower=0.0, upper=float(box_size) / 2.0)
    counts, _ = np.histogram(distances, bins=edges)
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    rho = len(points) / (float(box_size) ** 2)
    expected = 0.5 * len(points) * rho * shell_area
    g = np.divide(counts, expected, out=np.zeros_like(expected, dtype=np.float64), where=expected > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, g, counts.astype(np.float64)


def original_anisotropic_rdf(
    points: np.ndarray,
    box_size: float,
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vectors = _upper_pair_vectors_np(points, box_size, periodic=True)
    distances = np.sqrt(np.sum(vectors * vectors, axis=1))
    theta = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)

    r_edges = _centers_to_edges(r_grid, lower=0.0, upper=float(box_size) / 2.0)
    dtheta = np.pi / int(len(theta_grid))
    theta_index = np.floor((theta + 0.5 * dtheta) / dtheta).astype(int) % int(len(theta_grid))
    r_index = np.searchsorted(r_edges, distances, side="right") - 1
    valid = (r_index >= 0) & (r_index < len(r_edges) - 1)

    counts = np.zeros((len(r_edges) - 1, len(theta_grid)), dtype=np.float64)
    np.add.at(counts, (r_index[valid], theta_index[valid]), 1.0)

    annular_sector_area = (r_edges[1:] ** 2 - r_edges[:-1] ** 2)[:, None] * dtheta
    rho = len(points) / (float(box_size) ** 2)
    expected = 0.5 * len(points) * rho * annular_sector_area
    g = np.divide(counts, expected, out=np.zeros_like(counts), where=expected > 0)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    return r_centers, theta_grid.astype(np.float64), g, counts


def original_nnd_distribution(points: np.ndarray, box_size: float, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nnd = _nearest_distances_np(points, box_size, periodic=True)
    edges = _centers_to_edges(grid, lower=0.0)
    counts, _ = np.histogram(nnd, bins=edges)
    prob = counts.astype(np.float64) / max(1, len(nnd))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prob, nnd


def hard_periodic_voronoi_areas_grid(
    points: np.ndarray,
    box_size: float,
    resolution: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    points = _as_float_array(points)
    axis = (np.arange(int(resolution), dtype=np.float64) + 0.5) * float(box_size) / float(resolution)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    counts = np.zeros((len(points),), dtype=np.int64)
    for start in range(0, len(grid_points), int(chunk_size)):
        batch = grid_points[start : start + int(chunk_size)]
        delta = batch[:, None, :] - points[None, :, :]
        delta = delta - float(box_size) * np.round(delta / float(box_size))
        dist2 = np.sum(delta * delta, axis=2)
        nearest = np.argmin(dist2, axis=1)
        counts += np.bincount(nearest, minlength=len(points))
    cell_area = float(box_size) * float(box_size) / float(resolution * resolution)
    return counts.astype(np.float64) * cell_area


def original_voronoi_distribution(
    points: np.ndarray,
    box_size: float,
    grid: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    areas = hard_periodic_voronoi_areas_grid(points, box_size, resolution)
    edges = _centers_to_edges(grid, lower=0.0)
    counts, _ = np.histogram(areas, bins=edges)
    prob = counts.astype(np.float64) / max(1, len(areas))
    cv = float(np.std(areas) / np.mean(areas))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prob, areas, cv


def original_coordination_distribution(
    points: np.ndarray,
    box_size: float,
    grid: np.ndarray,
    cutoff_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((len(points),), dtype=np.int64)
    for i in range(len(points)):
        delta = points - points[i]
        delta = delta - float(box_size) * np.round(delta / float(box_size))
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        values[i] = int(np.count_nonzero((dist <= float(cutoff_radius)) & (dist > 0.0)))
    edges = _centers_to_edges(grid)
    counts, _ = np.histogram(values, bins=edges)
    prob = counts.astype(np.float64) / max(1, len(values))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prob, values.astype(np.float64)


def optimizer_consistent_statistics(
    points: np.ndarray,
    radius: float,
    box_size: float,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    norm_cfg = normalize_config(config, float(box_size))
    x = jnp.asarray(points / float(box_size), dtype=jnp.float32)
    stats = build_targets_from_reference(x, norm_cfg)
    arrays = {
        "rdf_grid": np.asarray(stats.rdf_grid) * float(box_size),
        "rdf_values": np.asarray(stats.rdf_values),
        "ardf_r_grid": np.asarray(stats.ardf_r_grid) * float(box_size),
        "ardf_theta_grid": np.asarray(stats.ardf_theta_grid),
        "ardf_values": np.asarray(stats.ardf_values),
        "nn_grid": np.asarray(stats.nn_grid) * float(box_size),
        "nn_values": np.asarray(stats.nn_values),
        "voronoi_area_grid": np.asarray(stats.voronoi_area_grid) * float(box_size) * float(box_size),
        "voronoi_area_values": np.asarray(stats.voronoi_area_values),
        "cv_value": np.asarray(stats.cv_value),
        "coord_grid": np.asarray(stats.coord_grid),
        "coord_values": np.asarray(stats.coord_values),
        "mean_coordination": np.asarray(stats.mean_coordination),
        "soft_nearest_neighbor_distances": np.asarray(
            soft_nearest_neighbor_distances(x, 1.0, norm_cfg["nearest_neighbor"]["tau"])
        )
        * float(box_size),
        "soft_coordination_numbers": np.asarray(
            smooth_coordination_numbers(
                x,
                1.0,
                norm_cfg["coordination"]["cutoff_radius"],
                norm_cfg["coordination"]["beta"],
            )
        ),
        "soft_voronoi_areas": np.asarray(
            soft_voronoi_areas(
                x,
                1.0,
                norm_cfg["cv"]["grid_resolution"],
                norm_cfg["cv"]["softmax_temperature"],
            )
        )
        * float(box_size)
        * float(box_size),
    }
    summary = {
        "optimizer_overlap_loss": float(overlap_loss(x, 1.0, float(radius) / float(box_size))),
        "soft_voronoi_cv": float(np.asarray(stats.cv_value)),
        "soft_mean_coordination": float(np.asarray(stats.mean_coordination)),
        "soft_nnd_mean": float(np.mean(arrays["soft_nearest_neighbor_distances"])),
    }
    return arrays, summary


def original_statistics(
    points: np.ndarray,
    radius: float,
    radii: np.ndarray | None,
    box_size: float,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rdf_cfg = config["rdf"]
    ardf_cfg = config["anisotropic_rdf"]
    nn_cfg = config["nearest_neighbor"]
    vor_cfg = config["voronoi_area_distribution"]
    cv_cfg = config["cv"]
    coord_cfg = config["coordination"]

    rdf_grid = np.linspace(float(rdf_cfg["r_min"]), float(rdf_cfg["r_max"]), int(rdf_cfg["num_bins"]))
    ardf_r_grid = np.linspace(float(ardf_cfg["r_min"]), float(ardf_cfg["r_max"]), int(ardf_cfg["num_r_bins"]))
    ardf_theta_grid = np.linspace(0.0, np.pi, int(ardf_cfg["num_theta_bins"]), endpoint=False)
    nn_grid = np.linspace(float(nn_cfg["grid_min"]), float(nn_cfg["grid_max"]), int(nn_cfg["num_grid"]))
    vor_grid = np.linspace(float(vor_cfg["grid_min"]), float(vor_cfg["grid_max"]), int(vor_cfg["num_grid"]))
    coord_grid = np.linspace(float(coord_cfg["grid_min"]), float(coord_cfg["grid_max"]), int(coord_cfg["num_grid"]))

    rdf_centers, rdf_values, rdf_counts = original_rdf(points, box_size, rdf_grid)
    ardf_r_centers, ardf_theta_centers, ardf_values, ardf_counts = original_anisotropic_rdf(
        points, box_size, ardf_r_grid, ardf_theta_grid
    )
    nn_centers, nn_prob, exact_nnd = original_nnd_distribution(points, box_size, nn_grid)
    vor_centers, vor_prob, vor_areas, vor_cv = original_voronoi_distribution(
        points, box_size, vor_grid, int(cv_cfg.get("grid_resolution", vor_cfg.get("grid_resolution", 64)))
    )
    coord_centers, coord_prob, coord_numbers = original_coordination_distribution(
        points, box_size, coord_grid, float(coord_cfg["cutoff_radius"])
    )

    arrays = {
        "rdf_grid": rdf_centers,
        "rdf_values": rdf_values,
        "rdf_counts": rdf_counts,
        "ardf_r_grid": ardf_r_centers,
        "ardf_theta_grid": ardf_theta_centers,
        "ardf_values": ardf_values,
        "ardf_counts": ardf_counts,
        "nn_grid": nn_centers,
        "nn_probability": nn_prob,
        "exact_nearest_neighbor_distances": exact_nnd,
        "voronoi_area_grid": vor_centers,
        "voronoi_area_probability": vor_prob,
        "hard_voronoi_areas": vor_areas,
        "coord_grid": coord_centers,
        "coord_probability": coord_prob,
        "exact_coordination_numbers": coord_numbers,
    }
    summary = {
        "hard_core_periodic_mean_radius": exact_hard_core_summary(points, radius, box_size, periodic=True),
        "hard_core_nonperiodic_mean_radius": exact_hard_core_summary(points, radius, box_size, periodic=False),
        "voronoi_cv": float(vor_cv),
        "exact_nnd_mean": float(np.mean(exact_nnd)),
        "exact_nnd_min": float(np.min(exact_nnd)),
        "exact_coordination_mean": float(np.mean(coord_numbers)),
    }
    if radii is not None:
        summary["hard_core_periodic_variable_radii"] = exact_hard_core_summary(points, radii, box_size, periodic=True)
        summary["hard_core_nonperiodic_variable_radii"] = exact_hard_core_summary(points, radii, box_size, periodic=False)
    return arrays, summary


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_definitions(path: Path) -> None:
    text = r"""# Particle Statistics Definitions

This report stores two versions of each statistic.

## Optimizer-consistent definitions

These arrays are computed by calling the same functions used by the optimization objective in `statistics`.

- Hard-core feasibility: `overlap_loss`, i.e. the half-sum of squared positive overlap depths under periodic minimum-image distances.
- RDF: smoothed pair-distance KDE divided by shell circumference and normalized so the discrete vector sums to one.
- Anisotropic RDF: smoothed pair-distance and axial pair-angle KDE, with pair angle modulo pi, shell-normalized and globally normalized.
- NND distribution: soft-min nearest-neighbor distance followed by normalized Gaussian KDE.
- Voronoi area distribution/CV: soft periodic Voronoi grid quadrature followed by normalized KDE; CV is `std(area)/mean(area)`.
- Coordination distribution: sigmoid-smoothed neighbor counts followed by normalized Gaussian KDE; the mean soft coordination is also saved.

## Classical/original definitions

- Hard-core feasibility: exact pairwise check `distance_ij >= r_i + r_j`; periodic versions use minimum-image distances.
- RDF \(g(r)\): exact pair-distance histogram normalized by the ideal-gas expected count in each annulus, \(0.5 N \rho \pi(r_{k+1}^2-r_k^2)\).
- Anisotropic RDF \(g(r,\theta)\): exact histogram of pair distance and axial pair angle \(\theta\in[0,\pi)\), normalized by the ideal-gas expected count in each annular angular sector.
- NND distribution: exact nearest-neighbor distances and a probability histogram on the configured grid.
- Voronoi area distribution/CV: hard periodic Voronoi assignment by nearest-center grid quadrature; the distribution is a probability histogram and CV is `std(area)/mean(area)`.
- Coordination-number distribution: exact count of neighbors within the configured cutoff radius and a probability histogram.
"""
    path.write_text(text, encoding="utf-8")


def compute_dataset_report(
    name: str,
    points: np.ndarray,
    radius: float,
    radii: np.ndarray | None,
    box_size: float,
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    opt_arrays, opt_summary = optimizer_consistent_statistics(points, radius, box_size, config)
    orig_arrays, orig_summary = original_statistics(points, radius, radii, box_size, config)
    _save_npz(out_dir / f"{name}_optimizer_consistent_stats.npz", opt_arrays)
    _save_npz(out_dir / f"{name}_original_stats.npz", orig_arrays)
    return {
        "optimizer_consistent": opt_summary,
        "original": orig_summary,
        "files": {
            "optimizer_consistent_npz": f"{name}_optimizer_consistent_stats.npz",
            "original_npz": f"{name}_original_stats.npz",
        },
    }


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def compare_saved_statistics(out_dir: Path) -> dict[str, Any]:
    det_opt = np.load(out_dir / "detected_optimizer_consistent_stats.npz")
    rve_opt = np.load(out_dir / "rve_optimizer_consistent_stats.npz")
    det_orig = np.load(out_dir / "detected_original_stats.npz")
    rve_orig = np.load(out_dir / "rve_original_stats.npz")

    return {
        "optimizer_consistent_mse": {
            "rdf": _mse(det_opt["rdf_values"], rve_opt["rdf_values"]),
            "anisotropic_rdf": _mse(det_opt["ardf_values"], rve_opt["ardf_values"]),
            "nnd_distribution": _mse(det_opt["nn_values"], rve_opt["nn_values"]),
            "voronoi_area_distribution": _mse(det_opt["voronoi_area_values"], rve_opt["voronoi_area_values"]),
            "coordination_distribution": _mse(det_opt["coord_values"], rve_opt["coord_values"]),
            "voronoi_cv_abs_diff": float(abs(float(det_opt["cv_value"]) - float(rve_opt["cv_value"]))),
            "mean_coordination_abs_diff": float(
                abs(float(det_opt["mean_coordination"]) - float(rve_opt["mean_coordination"]))
            ),
        },
        "original_mse": {
            "rdf": _mse(det_orig["rdf_values"], rve_orig["rdf_values"]),
            "anisotropic_rdf": _mse(det_orig["ardf_values"], rve_orig["ardf_values"]),
            "nnd_probability": _mse(det_orig["nn_probability"], rve_orig["nn_probability"]),
            "voronoi_area_probability": _mse(
                det_orig["voronoi_area_probability"], rve_orig["voronoi_area_probability"]
            ),
            "coordination_probability": _mse(det_orig["coord_probability"], rve_orig["coord_probability"]),
            "voronoi_cv_abs_diff": float(
                abs(
                    float(np.std(det_orig["hard_voronoi_areas"]) / np.mean(det_orig["hard_voronoi_areas"]))
                    - float(np.std(rve_orig["hard_voronoi_areas"]) / np.mean(rve_orig["hard_voronoi_areas"]))
                )
            ),
            "mean_coordination_abs_diff": float(
                abs(float(np.mean(det_orig["exact_coordination_numbers"])) - float(np.mean(rve_orig["exact_coordination_numbers"])))
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute optimizer-consistent and original particle statistics.")
    parser.add_argument("--config", type=Path, default=Path("configs/from_detected_fibers.yaml"))
    parser.add_argument("--detected", type=Path, default=Path("image_detection/result_files/fiber_centers_normalized.txt"))
    parser.add_argument("--rve", type=Path, default=Path("outputs/from_detected_fibers/rve_0000.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/statistics_report"))
    args = parser.parse_args()

    config = load_config(args.config)
    box_size = float(config["geometry"]["box_size"])
    geom = config["geometry"]
    base_radius = float(geom["radius"] if geom.get("radius") is not None else 0.5 * float(geom["diameter"]))
    radius_scale = float(
        geom.get(
            "statistics_radius_scale",
            geom.get("hard_core_radius_scale", geom.get("radius_scale", 1.0)),
        )
    )
    radius = base_radius * radius_scale

    detected = np.loadtxt(args.detected, dtype=np.float64)
    detected_points = detected[:, :2]
    detected_radii = detected[:, 2] * radius_scale if detected.shape[1] >= 3 else None

    rve_data = np.load(args.rve)
    rve_points = np.asarray(rve_data["points_final"], dtype=np.float64)
    rve_radius = float(np.asarray(rve_data["radius"]))
    rve_box = float(np.asarray(rve_data["box_size"])[0])

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_definitions(out_dir / "statistics_definitions.md")

    detected_report = compute_dataset_report(
        "detected",
        detected_points,
        radius,
        detected_radii,
        box_size,
        config,
        out_dir,
    )
    rve_report = compute_dataset_report(
        "rve",
        rve_points,
        rve_radius,
        None,
        rve_box,
        config,
        out_dir,
    )

    summary = {
        "config": str(args.config),
        "detected_source": str(args.detected),
        "rve_source": str(args.rve),
        "detected": detected_report,
        "rve": rve_report,
        "comparison": compare_saved_statistics(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
