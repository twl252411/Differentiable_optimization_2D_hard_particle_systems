"""Plot the four-panel matched-reference method comparison."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures"))

from plotting.manuscript_plot_style import (  # noqa: E402
    COLORS,
    TICK_LABEL_FONTSIZE,
    add_panel_labels,
    legend_upper_left,
    save_figure,
    setup_grid,
)


METHODS = ("overlap_only", "rdf_only", "rdf_ardf", "gradient_free", "proposed")
METHOD_LABELS = {
    "overlap_only": "OV",
    "rdf_only": "RO",
    "rdf_ardf": "RA",
    "gradient_free": "GF",
    "proposed": "PR",
}
TARGETS = ("image", "numerical")
TARGET_LABELS = {"image": "Image-based", "numerical": "Numerical"}
TARGET_COLORS = {"image": COLORS["gray"], "numerical": COLORS["blue"]}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_values(
    rows: list[dict[str, str]],
    target: str,
    method: str,
    key: str,
) -> np.ndarray:
    selected = [
        row for row in rows if row["target"] == target and row["method"] == method
    ]
    if key == "updates":
        return np.asarray(
            [
                float(row["stage1_iterations"]) + float(row["stage2_iterations"])
                for row in selected
            ]
        )
    return np.asarray([float(row[key]) for row in selected])


def log_tick_label(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10.0**exponent)
    if np.isclose(coefficient, 1.0):
        return "$" + rf"10^{{{exponent}}}" + "$"
    return "$" + rf"{coefficient:g}\times10^{{{exponent}}}" + "$"


def plot_panel(
    ax,
    rows: list[dict[str, str]],
    key: str,
    ylabel: str,
    *,
    log_scale: bool = False,
    scale: float = 1.0,
    limits: tuple[float, float] | None = None,
    ticks: tuple[float, ...] | None = None,
) -> None:
    x = np.arange(len(METHODS))
    width = 0.32
    for target_index, target in enumerate(TARGETS):
        means = [
            select_values(rows, target, method, key).mean() * scale
            for method in METHODS
        ]
        positions = x + (target_index - 0.5) * width
        ax.bar(
            positions,
            means,
            width=width,
            label=TARGET_LABELS[target],
            color=TARGET_COLORS[target],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in METHODS],
        rotation=0,
        ha="center",
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.60, 4.60)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
    if limits is not None:
        ax.set_ylim(*limits)
    if ticks is None:
        lower, upper = ax.get_ylim()
        ticks = tuple(
            np.geomspace(lower, upper, 5) if log_scale else np.linspace(lower, upper, 5)
        )
    ax.set_yticks(ticks)
    if log_scale:
        ax.set_yticklabels([log_tick_label(value) for value in ticks])
    legend_upper_left(ax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.runs_csv)
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
    plot_panel(
        axes[0, 0],
        rows,
        "runtime_s",
        "Runtime (s)",
        log_scale=True,
        limits=(1.0e-2, 1.0e2),
        ticks=(1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2),
    )
    plot_panel(
        axes[0, 1],
        rows,
        "conv_error_average",
        "Relative error",
        limits=(0.0, 1.0),
        ticks=(0.0, 0.25, 0.50, 0.75, 1.00),
    )
    axes[0, 1].set_yticklabels(("0", "0.25", "0.50", "0.75", "1.00"))
    plot_panel(
        axes[1, 0],
        rows,
        "min_normalized_gap",
        "Min normalized gap",
        log_scale=True,
        limits=(1.0e-5, 1.0e-1),
        ticks=(1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1),
    )
    plot_panel(
        axes[1, 1],
        rows,
        "updates",
        "Iterations to acceptance",
        log_scale=True,
        limits=(1.0e0, 1.0e4),
        ticks=(1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4),
    )
    add_panel_labels(axes.ravel(), label_offset=(-38, 8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
