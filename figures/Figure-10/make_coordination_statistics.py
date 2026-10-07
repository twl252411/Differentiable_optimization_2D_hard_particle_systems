"""Make coordination-statistics figure from the image-derived RVE result."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))

from _figure_common import compute_case_average_stage_data  # noqa: E402
from plotting.manuscript_plot_style import (  # noqa: E402
    AxisSpec,
    add_panel_labels,
    apply_axis_spec,
    legend_upper_right,
    save_figure,
    setup_grid,
)


FIGURE_NAME = "Figure-10"
CASES = ("image_reference", "random_reference")
AXIS_LABEL_FONTSIZE = 12.5
STANDARD_LINE_STYLES = (
    {"color": "#000000", "linestyle": "-", "linewidth": 1.2},
    {"color": "#FF0000", "linestyle": "--", "linewidth": 1.2},
    {"color": "#0000FF", "linestyle": "-.", "linewidth": 1.2},
)
COORDINATION_AXIS_SPECS = {
    "soft_x": AxisSpec(
        limits=(0.0, 12.0), tick_start=0.0, tick_stop=12.0, tick_step=3.0, decimals=1
    ),
    "soft_y": AxisSpec(
        limits=(0.00, 0.24), tick_start=0.00, tick_stop=0.24, tick_step=0.06, decimals=2
    ),
    "classical_x": AxisSpec(
        limits=(0.0, 12.0), tick_start=0.0, tick_stop=12.0, tick_step=3.0, decimals=1
    ),
    "classical_y": AxisSpec(
        limits=(0.00, 0.60), tick_start=0.00, tick_stop=0.60, tick_step=0.15, decimals=2
    ),
}


def _plot_coordination_row(
    axes: np.ndarray, stats: dict, axis_specs: dict[str, AxisSpec]
) -> None:
    target_o = stats["target"]["optimizer"]
    stage1_o = stats["stage1"]["optimizer"]
    final_o = stats["final"]["optimizer"]
    target_c = stats["target"]["original"]
    stage1_c = stats["stage1"]["original"]
    final_c = stats["final"]["original"]

    axes[0].plot(
        target_o["coord_grid"],
        target_o["coord_values"],
        label="Reference",
        **STANDARD_LINE_STYLES[0],
    )
    axes[0].plot(
        stage1_o["coord_grid"],
        stage1_o["coord_values"],
        label="Overlap-removal",
        **STANDARD_LINE_STYLES[1],
    )
    axes[0].plot(
        final_o["coord_grid"],
        final_o["coord_values"],
        label="Reconstruction",
        **STANDARD_LINE_STYLES[2],
    )
    axes[0].set_xlabel(r"$z$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel("Soft coordination")
    apply_axis_spec(axes[0], x=axis_specs["soft_x"], y=axis_specs["soft_y"])

    axes[1].plot(
        target_c["coord_grid"],
        target_c["coord_probability"],
        label="Reference",
        **STANDARD_LINE_STYLES[0],
    )
    axes[1].plot(
        stage1_c["coord_grid"],
        stage1_c["coord_probability"],
        label="Overlap-removal",
        **STANDARD_LINE_STYLES[1],
    )
    axes[1].plot(
        final_c["coord_grid"],
        final_c["coord_probability"],
        label="Reconstruction",
        **STANDARD_LINE_STYLES[2],
    )
    axes[1].set_xlabel(r"$z$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel("Probability")
    apply_axis_spec(axes[1], x=axis_specs["classical_x"], y=axis_specs["classical_y"])

    for ax in axes:
        legend_upper_right(ax)


def plot_combined_coordination(case_stats: dict[str, dict], local_dir: Path) -> None:
    fig, axes = setup_grid(
        2,
        2,
        figsize=(6.7, 5.85),
        left=0.105,
        right=0.965,
        bottom=0.095,
        top=0.965,
        wspace=0.15,
        hspace=0.275,
        square=True,
    )
    _plot_coordination_row(
        axes[0], case_stats["image_reference"], COORDINATION_AXIS_SPECS
    )
    _plot_coordination_row(
        axes[1], case_stats["random_reference"], COORDINATION_AXIS_SPECS
    )
    add_panel_labels(axes)
    save_figure(fig, local_dir / f"{FIGURE_NAME}.png")


def main() -> None:
    local_dir = Path(__file__).resolve().parent
    case_stats = {}
    for case in CASES:
        _, _, _, stats, _, _ = compute_case_average_stage_data(case, sample_count=10)
        case_stats[case] = stats
    plot_combined_coordination(case_stats, local_dir)


if __name__ == "__main__":
    main()
