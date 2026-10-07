"""Make convergence-history figure from the image-derived RVE result."""

from __future__ import annotations

import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))

from _figure_common import load_case_result  # noqa: E402
from rve_study.rve_study import COLORS, add_panel_labels, save_figure, setup_grid  # noqa: E402
from plotting.manuscript_plot_style import (
    AxisSpec,
    apply_axis_spec,
)  # noqa: E402


FIGURE_NAME = "Figure-4"
CASES = ("image_reference", "random_reference")
IMAGE_AXIS_SPECS = {
    # Panels (a,b): image-derived reference.
    "loss_x": AxisSpec(
        limits=(0.0, 36.0), ticks=(0.0, 9.0, 18.0, 27.0, 36.0), decimals=0
    ),
    "loss_y": AxisSpec(
        limits=(1.0e-9, 1.0e3), ticks=(1.0e-9, 1.0e-6, 1.0e-3, 1.0e0, 1.0e3)
    ),
    "overlap_y": AxisSpec(
        limits=(1.0e-16, 1.0e-8), ticks=(1.0e-16, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8)
    ),
}
NUMERICAL_AXIS_SPECS = {
    # Panels (c,d): random reference.
    "loss_x": AxisSpec(
        limits=(0.0, 40.0), ticks=(0.0, 10.0, 20.0, 30.0, 40.0), decimals=0
    ),
    "loss_y": AxisSpec(
        limits=(1.0e-9, 1.0e3), ticks=(1.0e-9, 1.0e-6, 1.0e-3, 1.0e0, 1.0e3)
    ),
    "overlap_y": AxisSpec(
        limits=(1.0e-16, 1.0e-8), ticks=(1.0e-16, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8)
    ),
}
CASE_AXIS_SPECS = {
    "image_reference": IMAGE_AXIS_SPECS,
    "random_reference": NUMERICAL_AXIS_SPECS,
}
FIGURE_LAYOUT = {
    "figsize": (6.7, 5.85),
    "left": 0.105,
    "right": 0.965,
    "bottom": 0.095,
    "top": 0.965,
    "wspace": 0.15,
    "hspace": 0.275,
}
STANDARD_LINE_STYLES = (
    {"color": "#000000", "linestyle": "-", "linewidth": 1.2},
    {"color": "#FF0000", "linestyle": (0, (9.0, 3.0)), "linewidth": 1.2},
    {"color": "#0000FF", "linestyle": "-.", "linewidth": 1.2},
    {"color": "#800080", "linestyle": (0, (3.0, 2.0)), "linewidth": 1.2},
    {
        "color": "#008000",
        "linestyle": (0, (7.0, 2.0, 1.5, 2.0, 1.5, 2.0)),
        "linewidth": 1.2,
    },
    {"color": "#8000FF", "linestyle": (0, (1.0, 1.6)), "linewidth": 1.2},
)


def legend_loss(ax) -> None:
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.94),
        ncol=3,
        borderaxespad=0.2,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
    )


def _plot_loss_history(ax, result, axis_specs: dict[str, AxisSpec]) -> None:
    hist = result.history
    if not hist:
        return
    it = [row["iteration"] for row in hist]
    ax.plot(
        it,
        [row["loss_total"] for row in hist],
        label="Total",
        **STANDARD_LINE_STYLES[0],
    )
    for idx, (key, label) in enumerate(
        [
            ("loss_rdf", "RDF"),
            ("loss_ardf", "ARDF"),
            ("loss_nn", "NND"),
            ("loss_voronoi", "Voronoi"),
            ("loss_coord", "Coord."),
        ],
        start=1,
    ):
        style = dict(STANDARD_LINE_STYLES[idx])
        ax.plot(it, [row[key] for row in hist], label=label, **style)
    ax.set_yscale("log")
    ax.set_xlabel("Stage-2 iteration")
    ax.set_ylabel("Loss")
    apply_axis_spec(ax, x=axis_specs["loss_x"], y=axis_specs["loss_y"])
    legend_loss(ax)


def _plot_overlap(ax, result, axis_specs: dict[str, AxisSpec]) -> None:
    potentials = [
        result.stage_metrics["stage1"].get(
            "overlap_loss", result.stage_metrics["stage1"].get("overlap_potential", 0.0)
        ),
        result.stage_metrics["stage2"].get(
            "final_overlap_loss", result.final_terms.get("loss_overlap", 0.0)
        ),
    ]
    y_min = (
        axis_specs["overlap_y"].limits[0]
        if axis_specs["overlap_y"].limits is not None
        else 1.0e-16
    )
    plot_potentials = [max(value, y_min) for value in potentials]
    x = [0.35, 0.65]
    ax.bar(
        x,
        [value - y_min for value in plot_potentials],
        bottom=y_min,
        width=0.175,
        color=[COLORS["blue"], COLORS["orange"]],
    )
    ax.set_xlim(0.15, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stage 1", "Stage 2"])
    ax.set_yscale("log")
    ax.set_ylabel("Overlap potential")
    apply_axis_spec(ax, y=axis_specs["overlap_y"])


def plot_combined_convergence(results: dict[str, object], fig_dir: Path) -> None:
    fig, axes = setup_grid(
        2,
        2,
        figsize=FIGURE_LAYOUT["figsize"],
        left=FIGURE_LAYOUT["left"],
        right=FIGURE_LAYOUT["right"],
        bottom=FIGURE_LAYOUT["bottom"],
        top=FIGURE_LAYOUT["top"],
        wspace=FIGURE_LAYOUT["wspace"],
        hspace=FIGURE_LAYOUT["hspace"],
        square=True,
    )
    for row, case in enumerate(CASES):
        result = results[case]
        axis_specs = CASE_AXIS_SPECS[case]
        _plot_loss_history(axes[row, 0], result, axis_specs)
        _plot_overlap(axes[row, 1], result, axis_specs)
        # axes[row, 0].set_title(case_labels[case], fontsize=PANEL_TITLE_FONTSIZE, pad=3.0)
        # axes[row, 1].set_title(case_labels[case], fontsize=PANEL_TITLE_FONTSIZE, pad=3.0)
    add_panel_labels(axes, label_offset=(-43, 8))
    save_figure(fig, fig_dir / f"{FIGURE_NAME}.png")


def main() -> None:
    local_dir = Path(__file__).resolve().parent
    results = {}
    for case in CASES:
        result = load_case_result(case)
        results[case] = result
    plot_combined_convergence(results, local_dir)


if __name__ == "__main__":
    main()
