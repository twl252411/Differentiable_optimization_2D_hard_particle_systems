"""Shared manuscript-style matplotlib helpers.

The settings mirror the visual conventions used by
`docs/prl_manuscript/scripts/make_manuscript_figures.py` in the companion
repository: serif fonts, inward four-sided ticks, external panel labels, and
compact manually controlled subplot spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


TICK_LABEL_FONTSIZE = 10
PANEL_TITLE_FONTSIZE = 10.5
Y_MAJOR_TICK_COUNT = 5

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": PANEL_TITLE_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": 10,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#666666",
    "light_blue": "#D6EAF8",
    "light_orange": "#FCE5CD",
}


@dataclass(frozen=True)
class AxisSpec:
    """Configurable axis limits, ticks, and numeric tick-label precision."""

    limits: tuple[float, float] | None = None
    tick_start: float | None = None
    tick_stop: float | None = None
    tick_step: float | None = None
    ticks: tuple[float, ...] | list[float] | None = None
    tick_labels: tuple[str, ...] | list[str] | None = None
    decimals: int | None = None


def make_ticks(start: float, stop: float, step: float) -> np.ndarray:
    return np.arange(start, stop + 0.5 * step, step)


def _configured_ticks(spec: AxisSpec):
    if spec.ticks is not None:
        return list(spec.ticks)
    if (
        spec.tick_start is not None
        and spec.tick_stop is not None
        and spec.tick_step is not None
    ):
        return make_ticks(spec.tick_start, spec.tick_stop, spec.tick_step)
    return None


def _five_y_ticks(ax: plt.Axes, spec: AxisSpec) -> tuple[np.ndarray, list[str] | None]:
    """Return exactly five major y ticks without changing the axis limits."""

    configured = _configured_ticks(spec)
    if configured is not None and len(configured) == Y_MAJOR_TICK_COUNT:
        labels = list(spec.tick_labels) if spec.tick_labels is not None else None
        return np.asarray(configured, dtype=float), labels

    lower, upper = spec.limits if spec.limits is not None else ax.get_ylim()
    if ax.get_yscale() == "log":
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("logarithmic y limits must be positive")
        ticks = np.geomspace(lower, upper, Y_MAJOR_TICK_COUNT)
    else:
        ticks = np.linspace(lower, upper, Y_MAJOR_TICK_COUNT)

    labels = None
    if spec.tick_labels is not None:
        try:
            decimals = max(
                len(label.split(".", 1)[1]) if "." in label else 0
                for label in spec.tick_labels
            )
            labels = [f"{value:.{decimals}f}" for value in ticks]
        except (AttributeError, ValueError):
            labels = None
    return ticks, labels


def apply_axis_spec(
    ax: plt.Axes, *, x: AxisSpec | None = None, y: AxisSpec | None = None
) -> None:
    """Apply configurable x/y limits, ticks, labels, and decimal formatting."""

    for axis_name, spec in (("x", x), ("y", y)):
        if spec is None:
            continue
        axis = ax.xaxis if axis_name == "x" else ax.yaxis
        if spec.limits is not None:
            (ax.set_xlim if axis_name == "x" else ax.set_ylim)(*spec.limits)
        labels = list(spec.tick_labels) if spec.tick_labels is not None else None
        if axis_name == "y":
            ticks, labels = _five_y_ticks(ax, spec)
        else:
            ticks = _configured_ticks(spec)
        if ticks is not None:
            (ax.set_xticks if axis_name == "x" else ax.set_yticks)(ticks)
        if labels is not None:
            (ax.set_xticklabels if axis_name == "x" else ax.set_yticklabels)(labels)
            for label in axis.get_ticklabels():
                label.set_fontsize(TICK_LABEL_FONTSIZE)
        elif spec.decimals is not None:
            axis.set_major_formatter(FormatStrFormatter(f"%.{spec.decimals}f"))


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(
        direction="in", top=True, right=True, width=0.8, labelsize=TICK_LABEL_FONTSIZE
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def style_axes(axes) -> None:
    for ax in np.ravel(axes):
        style_axis(ax)


def legend_upper_right(ax: plt.Axes) -> None:
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
        handlelength=2.0,
    )


def legend_upper_left(ax: plt.Axes) -> None:
    ax.legend(frameon=False, loc="upper left", borderaxespad=0.0, handlelength=2.0)


def add_panel_labels(axes, labels=None, label_offset=(-38, 8)) -> None:
    axes = np.ravel(axes)
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        ax.annotate(
            lab,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=label_offset,
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=12.5,
            color="black",
            zorder=20,
            clip_on=False,
        )


def add_external_colorbar(
    fig: plt.Figure, ax: plt.Axes, mappable, label: str | None = None
):
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.010, pos.y0, 0.014, pos.height])
    cbar = fig.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(direction="in", width=0.8, length=3.0)
    return cbar


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 600) -> None:
    fig.canvas.draw()
    for index, ax in enumerate(fig.axes):
        if not ax.axison or ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar"):
            continue
        ticks = ax.get_yticks()
        labels = [
            label.get_text() for label in ax.get_yticklabels() if label.get_visible()
        ]
        if (
            len(ticks) != Y_MAJOR_TICK_COUNT
            or len(labels) != Y_MAJOR_TICK_COUNT
            or any(not label for label in labels)
        ):
            raise RuntimeError(
                f"{path.name}: axis {index} must have exactly {Y_MAJOR_TICK_COUNT} visible y ticks; "
                f"found {len(ticks)} ticks and {len([label for label in labels if label])} labels"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def setup_grid(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float],
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float,
    hspace: float,
    square: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )
    for ax in np.ravel(axes):
        if square:
            ax.set_box_aspect(1.0)
        style_axis(ax)
    return fig, axes
