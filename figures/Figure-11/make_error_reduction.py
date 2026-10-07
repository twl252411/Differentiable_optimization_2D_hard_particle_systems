"""Make descriptor-error reduction figure from the image-derived RVE result."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))

from _figure_common import compute_case_average_stage_data  # noqa: E402
from plotting.manuscript_plot_style import (
    AxisSpec,
    COLORS,
    TICK_LABEL_FONTSIZE,
    add_panel_labels,
    apply_axis_spec,
    legend_upper_left,
    save_figure,
    setup_grid,
)  # noqa: E402


FIGURE_NAME = "Figure-11"
CASES = ("image_reference", "random_reference")
PANEL_SIZE = 5.85 * (0.965 - 0.095) / (2.0 + 0.275)
COMBINED_FIGSIZE = (
    PANEL_SIZE * (2.0 + 0.15) / (0.965 - 0.105),
    PANEL_SIZE / (0.95 - 0.27),
)
DESCRIPTOR_AXIS_SPECS = {
    "descriptor_x": AxisSpec(limits=(-0.60, 6.60)),
    "descriptor_y": AxisSpec(
        limits=(1.0e-7, 1.0e1), ticks=(1.0e-7, 1.0e-5, 1.0e-3, 1.0e-1, 1.0e1)
    ),
}
DESCRIPTOR_LABELS = {
    "RDF": "RD",
    "ARDF": "AD",
    "NND": "NN",
    "Voronoi": "VA",
    "CV": "VC",
    "Coord.": "CN",
    "Mean coord.": "MC",
}


def _plot_error_bars(
    ax,
    errors: dict[str, dict[str, float]],
    axis_specs: dict[str, AxisSpec],
) -> None:
    labels = list(next(iter(errors.values())).keys())
    short_labels = [DESCRIPTOR_LABELS.get(label, label) for label in labels]
    x = np.arange(len(labels))
    width = 0.32
    stage_labels = {"stage1": "Stage-1", "stage2": "Stage-2"}
    for i, stage in enumerate(("stage1", "stage2")):
        vals = [errors[stage][label] for label in labels]
        color = [COLORS["gray"], COLORS["blue"]][i]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=stage_labels[stage],
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        short_labels, rotation=0, ha="center", fontsize=TICK_LABEL_FONTSIZE
    )
    ax.set_ylabel("Descriptor error")
    apply_axis_spec(ax, x=axis_specs["descriptor_x"], y=axis_specs["descriptor_y"])
    legend_upper_left(ax)


def plot_combined_errors(case_errors: dict[str, dict], local_dir: Path) -> None:
    fig, axes = setup_grid(
        1,
        2,
        figsize=COMBINED_FIGSIZE,
        left=0.105,
        right=0.965,
        bottom=0.27,
        top=0.95,
        wspace=0.40,
        hspace=0.0,
        square=True,
    )
    _plot_error_bars(axes[0], case_errors["image_reference"], DESCRIPTOR_AXIS_SPECS)
    _plot_error_bars(axes[1], case_errors["random_reference"], DESCRIPTOR_AXIS_SPECS)
    add_panel_labels(axes)
    save_figure(fig, local_dir / f"{FIGURE_NAME}.png")


def main() -> None:
    local_dir = Path(__file__).resolve().parent
    case_errors = {}
    for case in CASES:
        _, _, _, _, errors, _ = compute_case_average_stage_data(case, sample_count=10)
        case_errors[case] = errors
    plot_combined_errors(case_errors, local_dir)


if __name__ == "__main__":
    main()
