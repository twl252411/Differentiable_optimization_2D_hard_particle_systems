"""Generate Figure 5: configuration evolution for both reference cases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))

from _figure_common import MICRO_IMAGE, load_case_result, load_detected_points, require  # noqa: E402
from plotting.manuscript_plot_style import (
    AxisSpec,
    COLORS,
    PANEL_TITLE_FONTSIZE,
    add_panel_labels,
    apply_axis_spec,
    plt,
    save_figure,
    style_axis,
)  # noqa: E402


FIGURE_NAME = "Figure-5"
AXIS_LABEL_FONTSIZE = 12.5
CONFIG_AXIS_SPEC = AxisSpec(
    limits=(0.0, 1.0),
    ticks=(0.0, 0.25, 0.50, 0.75, 1.0),
    tick_labels=("0.00", "0.25", "0.50", "0.75", "1.00"),
)


def setup_figure2_panels():
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 6.75), constrained_layout=False)
    fig.subplots_adjust(
        left=0.10, right=0.965, bottom=0.18, top=0.94, wspace=0.125, hspace=0.325
    )
    for ax in axes.ravel():
        ax.set_box_aspect(1.0)
        style_axis(ax)
    return fig, axes


def set_config_axis(ax, box: float) -> None:
    ax.set_aspect("equal", adjustable="box")
    axis_spec = AxisSpec(
        limits=(CONFIG_AXIS_SPEC.limits[0] * box, CONFIG_AXIS_SPEC.limits[1] * box),
        ticks=tuple(value * box for value in CONFIG_AXIS_SPEC.ticks),
        tick_labels=CONFIG_AXIS_SPEC.tick_labels,
    )
    apply_axis_spec(ax, x=axis_spec, y=axis_spec)
    ax.set_xlabel(r"$x/L$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$y/L$", fontsize=AXIS_LABEL_FONTSIZE)


def draw_particles(ax, points: np.ndarray, radii, box: float) -> None:
    points = np.asarray(points, dtype=float)
    if np.isscalar(radii):
        radii = np.full(len(points), float(radii), dtype=float)
    radii = np.asarray(radii, dtype=float)
    for (x, y), radius in zip(points, radii):
        circle = plt.Circle(
            (x, y), radius, facecolor="none", edgecolor=COLORS["red"], linewidth=0.55
        )
        ax.add_patch(circle)
    ax.plot(points[:, 0], points[:, 1], ".", color=COLORS["blue"], markersize=1.2)
    set_config_axis(ax, box)


def flip_image_y(points: np.ndarray, box: float) -> np.ndarray:
    points = np.asarray(points, dtype=float).copy()
    points[:, 1] = box - points[:, 1]
    return points


def draw_micro_image(ax, points: np.ndarray, radii: np.ndarray, box: float) -> None:
    image = plt.imread(require(MICRO_IMAGE))
    short_side = min(image.shape[:2])
    image_square = image[:short_side, :short_side]
    ax.imshow(image_square, cmap="gray", extent=(0.0, box, 0.0, box), origin="upper")
    for (x, y), radius in zip(points, radii):
        circle = plt.Circle(
            (x, y), radius, facecolor="none", edgecolor="red", linewidth=0.45
        )
        ax.add_patch(circle)
    ax.plot(points[:, 0], points[:, 1], ".", color=COLORS["blue"], markersize=1.0)
    set_config_axis(ax, box)


def main() -> None:
    local_dir = Path(__file__).resolve().parent
    result = load_case_result("image_reference")
    detected_points, detected_radii = load_detected_points()
    box = float(np.asarray(result.box_size)[0])
    radius = float(getattr(result, "plot_radius", result.radius))
    detected_plot_points = flip_image_y(detected_points, box)

    fig, axes = setup_figure2_panels()
    panels = [
        (axes[0, 0], result.points_init, radius, "Initial"),
        (axes[0, 1], result.points_stage1, radius, "Stage-1"),
        (axes[1, 0], result.points_stage2, radius, "Stage-2"),
        (axes[1, 1], detected_plot_points, detected_radii, "Reference RVE"),
    ]
    for ax, points, radii, title in panels:
        draw_particles(ax, points, radii, box)
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    add_panel_labels(axes.ravel())
    save_figure(fig, local_dir / f"{FIGURE_NAME}.png")


if __name__ == "__main__":
    main()
