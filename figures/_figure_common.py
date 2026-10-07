"""Shared data loaders for the manuscript figure scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "figures"
COMMON_DATA_ROOT = FIGURE_ROOT / "data"
CASE_DIRS = {
    "image_reference": COMMON_DATA_ROOT / "reference_results" / "image_reference",
    "random_reference": COMMON_DATA_ROOT / "reference_results" / "random_reference",
}
DETECTED_POINTS = (
    ROOT / "image_detection" / "result_files" / "fiber_centers_normalized.txt"
)
MICRO_IMAGE = ROOT / "image_detection" / "fiber_composites_section_v1.jpg"


def require(path: Path) -> Path:
    """Return an existing path or raise a clear error."""
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _namespace_from_result_files(rve_path: Path, summary_path: Path) -> SimpleNamespace:
    with np.load(require(rve_path)) as data:
        values = {key: np.asarray(data[key]) for key in data.files}

    summary = json.loads(require(summary_path).read_text(encoding="utf-8"))
    radius = float(np.asarray(values["radius"]))
    return SimpleNamespace(
        points_init=np.asarray(values["points_init"], dtype=float),
        points_stage1=np.asarray(values["points_stage1"], dtype=float),
        points_stage2=np.asarray(values["points_stage2"], dtype=float),
        points_final=np.asarray(values["points_final"], dtype=float),
        points_target=np.asarray(values["points_target"], dtype=float),
        box_size=np.asarray(values["box_size"], dtype=float),
        radius=radius,
        plot_radius=float(np.asarray(values.get("plot_radius", radius))),
        plot_diameter=float(np.asarray(values.get("plot_diameter", 2.0 * radius))),
        stage_metrics=summary.get("stage_metrics", {}),
        final_terms=summary.get("final_terms", {}),
        history=summary.get("history", []),
    )


def load_case_result(case: str, index: int = 0) -> SimpleNamespace:
    directory = CASE_DIRS[case]
    return _namespace_from_result_files(
        directory / f"rve_{index:04d}.npz",
        directory / f"rve_{index:04d}_summary.json",
    )


def load_detected_points(path: Path = DETECTED_POINTS) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(require(path), dtype=float)
    points = np.asarray(data[:, :2], dtype=float)
    if data.shape[1] >= 3:
        radii = np.asarray(data[:, 2], dtype=float)
    else:
        radii = np.full(len(points), 0.21681174785100282)
    return points, radii


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(require(path)) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _mean_std(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    means: dict[str, Any] = {}
    standard_deviations: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], dict):
            means[key], standard_deviations[key] = _mean_std(values)
        else:
            stack = np.stack([np.asarray(value, dtype=float) for value in values])
            means[key] = np.mean(stack, axis=0)
            standard_deviations[key] = np.std(stack, axis=0, ddof=0)
    return means, standard_deviations


def compute_case_average_stage_data(
    case: str,
    *,
    sample_count: int = 10,
    sample_indices: list[int] | None = None,
):
    indices = (
        list(range(sample_count)) if sample_indices is None else list(sample_indices)
    )
    case_directory = require(COMMON_DATA_ROOT / case)
    sample_stats = []
    sample_errors = []

    for index in indices:
        sample_directory = require(case_directory / f"sample_{index:04d}")
        stages = {}
        for stage in ("target", "stage1", "final"):
            stages[stage] = {
                "points": np.empty((0, 2)),
                "optimizer": _load_npz(
                    sample_directory / f"{stage}_optimizer_consistent_stats.npz"
                ),
                "original": _load_npz(sample_directory / f"{stage}_original_stats.npz"),
            }
        sample_stats.append(stages)
        sample_errors.append(
            json.loads(
                (sample_directory / "descriptor_errors.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    aggregated = {}
    for stage in ("target", "stage1", "final"):
        optimizer_mean, optimizer_std = _mean_std(
            [sample[stage]["optimizer"] for sample in sample_stats]
        )
        original_mean, original_std = _mean_std(
            [sample[stage]["original"] for sample in sample_stats]
        )
        aggregated[stage] = {
            "points": np.empty((0, 2)),
            "optimizer": optimizer_mean,
            "optimizer_std": optimizer_std,
            "original": original_mean,
            "original_std": original_std,
        }

    mean_errors = {}
    std_errors = {}
    for stage in sample_errors[0]:
        mean_errors[stage] = {}
        std_errors[stage] = {}
        for name in sample_errors[0][stage]:
            values = np.asarray(
                [sample[stage][name] for sample in sample_errors], dtype=float
            )
            mean_errors[stage][name] = float(values.mean())
            std_errors[stage][name] = float(values.std(ddof=0))

    return {}, [], sample_stats, aggregated, mean_errors, std_errors


def copy_figure_files(
    source_dir: Path,
    source_name: str,
    target_dir: Path,
    target_name: str,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".svg", ".png"):
        source = require(source_dir / f"{source_name}{extension}")
        shutil.copy2(source, target_dir / f"{target_name}{extension}")


def remove_figure_files(directory: Path, name: str) -> None:
    for extension in (".pdf", ".svg", ".png"):
        path = directory / f"{name}{extension}"
        if path.exists():
            path.unlink()
