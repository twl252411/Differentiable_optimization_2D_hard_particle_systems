"""Plot matched-reference initialization sensitivity in the original four-panel layout."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator

FIGURES_DIR = Path(__file__).resolve().parents[1]
ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(FIGURES_DIR))

from plotting.manuscript_plot_style import (  # noqa: E402
    AxisSpec,
    COLORS,
    TICK_LABEL_FONTSIZE,
    add_panel_labels,
    apply_axis_spec,
    legend_upper_left,
    save_figure,
    setup_grid,
)

FIGURE_NAME = "Figure-3"
SUMMARY_CSV = (
    ROOT
    / "manuscript_data"
    / "aggregates"
    / "initialization_sensitivity_20_summary.csv"
)
ACCEPTANCE_CSV = (
    ROOT / "manuscript_data" / "aggregates" / "initialization_acceptance_audit.csv"
)
LAYOUT = {
    "figsize": (6.7, 5.85),
    "left": 0.105,
    "right": 0.965,
    "bottom": 0.095,
    "top": 0.965,
    "wspace": 0.15,
    "hspace": 0.275,
}
METHOD_ORDER = ("Regular", "Poisson", "Sobol", "Thomas")
REFERENCE_ORDER = ("Image-derived target", "Matched numerical target")
REFERENCE_LABELS = {
    "Image-derived target": "Image-based",
    "Matched numerical target": "Numerical",
}
REFERENCE_COLORS = {
    "Image-derived target": COLORS["gray"],
    "Matched numerical target": COLORS["blue"],
}
BAR_WIDTH = 0.35
AXIS_SPECS = {
    "method_x": AxisSpec(limits=(-0.60, 3.60)),
    "runtime_y": AxisSpec(
        limits=(0.0, 40.0),
        ticks=(0.0, 10.0, 20.0, 30.0, 40.0),
        tick_labels=("0.00", "10.0", "20.0", "30.0", "40.0"),
    ),
    "stage1_iter_y": AxisSpec(
        limits=(0.0, 4000.0), ticks=(0.0, 1000.0, 2000.0, 3000.0, 4000.0)
    ),
    "stage2_iter_y": AxisSpec(
        limits=(0.0, 100.0), ticks=(0.0, 25.0, 50.0, 75.0, 100.0)
    ),
    "final_objective_y": AxisSpec(
        limits=(1.0e-7, 1.0e-3), ticks=(1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    ),
}
PANEL_SPECS = (
    ("runtime_mean_s", "runtime_std_s", "Runtime (s)", "runtime_y", False),
    (
        "stage1_iter_mean",
        "stage1_iter_std",
        r"Stage-1 iterations ($\times 10^{3}$)",
        "stage1_iter_y",
        False,
    ),
    (
        "stage2_iter_mean",
        "stage2_iter_std",
        "Stage-2 iterations",
        "stage2_iter_y",
        False,
    ),
    ("final_loss_mean", "final_loss_std", "Final objective", "final_objective_y", True),
)


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_acceptance() -> list[dict[str, str]]:
    with ACCEPTANCE_CSV.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def grouped_bars(ax, rows, value_col, std_col, ylabel, axis_key, log_scale=False):
    lookup = {(row["case"], row["initialization"]): row for row in rows}
    x = np.arange(len(METHOD_ORDER), dtype=float)
    for index, reference in enumerate(REFERENCE_ORDER):
        values = np.asarray(
            [float(lookup[(reference, method)][value_col]) for method in METHOD_ORDER]
        )
        deviations = np.asarray(
            [float(lookup[(reference, method)][std_col]) for method in METHOD_ORDER]
        )
        ax.bar(
            x + (index - 0.5) * BAR_WIDTH,
            values,
            width=BAR_WIDTH,
            label=REFERENCE_LABELS[reference],
            color=REFERENCE_COLORS[reference],
            yerr=deviations,
            error_kw={
                "ecolor": "black",
                "elinewidth": 0.7,
                "capsize": 2.0,
                "capthick": 0.7,
            },
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        METHOD_ORDER, rotation=0, ha="center", fontsize=TICK_LABEL_FONTSIZE
    )
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
    apply_axis_spec(ax, x=AXIS_SPECS["method_x"], y=AXIS_SPECS[axis_key])
    if axis_key == "stage1_iter_y":
        ticks = np.linspace(0.0, 4000.0, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{value / 1000.0:.2f}" for value in ticks])
    legend_upper_left(ax)


def main() -> None:
    rows = load_rows()
    acceptance = load_acceptance()
    if len(rows) != 8 or len(acceptance) != 8:
        raise ValueError("expected eight 20-run groups")
    fig, axes = setup_grid(2, 2, square=True, **LAYOUT)
    for ax, (value_col, std_col, ylabel, axis_key, log_scale) in zip(
        axes.ravel(), PANEL_SPECS
    ):
        grouped_bars(ax, rows, value_col, std_col, ylabel, axis_key, log_scale)
    exceptions = [row for row in acceptance if row["joint_accepted"] != row["attempts"]]
    if exceptions:
        lines = [
            "Accepted: "
            + ", ".join(
                f"{row['target']} {row['initialization']}: {row['joint_accepted']}/{row['attempts']}"
                for row in exceptions
            )
        ]
    else:
        lines = ["Accepted: 20/20 for all groups"]
    axes[1, 1].text(
        0.98,
        0.94,
        "\n".join(lines),
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    add_panel_labels(axes, label_offset=(-45, 8))
    save_figure(fig, Path(__file__).resolve().parent / f"{FIGURE_NAME}.png")


if __name__ == "__main__":
    main()
