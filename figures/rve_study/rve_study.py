"""Plotting primitives used by the manuscript statistical figures."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matplotlib.ticker import FormatStrFormatter, FuncFormatter, ScalarFormatter
from plotting.manuscript_plot_style import (
    AxisSpec,
    COLORS,
    PANEL_TITLE_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    add_external_colorbar,
    add_panel_labels,
    apply_axis_spec,
    legend_upper_left,
    legend_upper_right,
    plt,
    save_figure,
    setup_grid,
    style_axis,
)
from statistics_report import optimizer_consistent_statistics, original_statistics


class FixedDecimalScalarFormatter(ScalarFormatter):
    def __init__(self, decimals: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = decimals

    def _set_format(self) -> None:
        self.format = f"%1.{self.decimals}f"
        if self._usetex or self._useMathText:
            self.format = rf"$\mathdefault{{{self.format}}}$"


AXIS_LABEL_FONTSIZE = 12.5
FOUR_PANEL_LAYOUT = {
    "figsize": (10.4869565, 5.85),
    "left": 0.105,
    "right": 0.90,
    "bottom": 0.095,
    "top": 0.965,
    "wspace": 0.30,
    "hspace": 0.275,
}
CONFIG_AXIS_SPEC = AxisSpec(
    limits=(0.0, 1.0),
    ticks=(0.0, 0.25, 0.50, 0.75, 1.0),
    tick_labels=("0.00", "0.25", "0.50", "0.75", "1.00"),
)
PAIR_AXIS_SPECS = {
    "search_radius_image": AxisSpec(
        limits=(0.5, 4.5), tick_start=0.5, tick_stop=4.5, tick_step=1.0, decimals=1
    ),
    "search_radius_random": AxisSpec(
        limits=(1.0, 6.0), tick_start=1.0, tick_stop=6.0, tick_step=1.0, decimals=1
    ),
    "search_radius_auto": AxisSpec(
        limits=(0.5, 5.0), tick_start=0.5, tick_stop=5.0, tick_step=1.0, decimals=1
    ),
    "theta": AxisSpec(
        limits=(0.0, np.pi),
        ticks=(0.0, 0.25 * np.pi, 0.50 * np.pi, 0.75 * np.pi, np.pi),
        tick_labels=(r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"),
    ),
    "soft_rdf_image": AxisSpec(
        limits=(0.0, 0.04), tick_start=0.0, tick_stop=0.04, tick_step=0.01, decimals=2
    ),
    "soft_rdf_random": AxisSpec(
        limits=(0.0, 0.08), tick_start=0.0, tick_stop=0.08, tick_step=0.02, decimals=2
    ),
    "classical_rdf": AxisSpec(
        limits=(0.0, 2.0), tick_start=0.0, tick_stop=2.0, tick_step=0.5, decimals=2
    ),
}
PAIR_COLORBAR_SPECS = {
    "image_ardf": {
        "vmin": 0.8e-3,
        "vmax": 2.4e-3,
        "ticks": np.arange(0.8, 2.4 + 1.0e-9, 0.4) * 1.0e-3,
    },
    "image_soft_error": {
        "vmax": 4.0e-4,
        "ticks": np.arange(0.0, 4.0 + 1.0e-9, 1.0) * 1.0e-4,
        "tick_scale": 1.0e-4,
        "tick_decimals": 1,
        "offset_text": r"$\times 10^{-4}$",
    },
    "image_classical_error": {
        "vmax": 4.0e-1,
        "ticks": np.arange(0.0, 4.0 + 1.0e-9, 1.0) * 1.0e-1,
        "tick_scale": 1.0e-1,
        "tick_decimals": 1,
        "offset_text": r"$\times 10^{-1}$",
    },
    "random_ardf": {
        "vmin": 1.0e-3,
        "vmax": 5.0e-3,
        "ticks": np.arange(1.0, 5.0 + 1.0e-9, 1.0) * 1.0e-3,
        "scientific_decimals": 1,
    },
    "random_soft_error": {
        "vmax": 1.2e-3,
        "ticks": np.arange(0.0, 1.2 + 1.0e-9, 0.3) * 1.0e-3,
    },
    "random_classical_error": {
        "vmax": 0.8,
        "ticks": np.arange(0.0, 0.8 + 1.0e-9, 0.2),
        "tick_decimals": 1,
    },
}
LOCAL_AXIS_SPECS = {
    "soft_nnd_x": AxisSpec(decimals=2),
    "soft_nnd_y": AxisSpec(decimals=2),
    "classical_nnd_x": AxisSpec(decimals=2),
    "classical_nnd_y": AxisSpec(decimals=2),
    "soft_voronoi_x": AxisSpec(decimals=2),
    "soft_voronoi_y": AxisSpec(decimals=2),
    "classical_voronoi_x": AxisSpec(decimals=2),
    "classical_voronoi_y": AxisSpec(decimals=2),
}
COORD_AXIS_SPECS = {
    "soft_x": AxisSpec(decimals=0),
    "soft_y": AxisSpec(decimals=2),
    "classical_x": AxisSpec(decimals=0),
    "classical_y": AxisSpec(decimals=2),
}
ERROR_AXIS_SPECS = {
    "descriptor_x": AxisSpec(),
    "descriptor_y": AxisSpec(),
}
CONVERGENCE_AXIS_SPECS = {
    "loss_x": AxisSpec(decimals=0),
    "loss_y": AxisSpec(),
    "overlap_y": AxisSpec(),
}
COST_AXIS_SPECS = {
    "runtime_x": AxisSpec(),
    "runtime_y": AxisSpec(decimals=0),
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
DESCRIPTOR_LABELS = {
    "RDF": "RD",
    "ARDF": "AD",
    "NND": "NN",
    "Voronoi": "VA",
    "CV": "VC",
    "Coord.": "CN",
    "Mean coord.": "MC",
}


def apply_standard_line_style(index: int, *, alpha: float = 1.0) -> dict[str, Any]:
    style = dict(STANDARD_LINE_STYLES[index])
    style["alpha"] = alpha
    return style


def plot_mean_curve(
    ax,
    x: np.ndarray,
    mean_y: np.ndarray,
    *,
    label: str,
    style: dict[str, Any],
    std_y: np.ndarray | None = None,
) -> None:
    ax.plot(x, mean_y, label=label, **style)


def draw_disks(ax, points: np.ndarray, radius: float, box: float, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    axis_spec = AxisSpec(
        limits=(0.0, box),
        ticks=tuple(value * box for value in CONFIG_AXIS_SPEC.ticks),
        tick_labels=CONFIG_AXIS_SPEC.tick_labels,
    )
    apply_axis_spec(ax, x=axis_spec, y=axis_spec)
    ax.set_xlabel(r"$x/L_x$")
    ax.set_ylabel(r"$y/L_y$")
    for x, y in points:
        circle = plt.Circle(
            (x, y),
            radius,
            facecolor=COLORS["red"],
            edgecolor=COLORS["red"],
            alpha=0.72,
            linewidth=0.0,
        )
        ax.add_patch(circle)
    style_axis(ax)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return float(np.mean((aa - bb) ** 2))


def save_stats_npz(
    out_dir: Path, prefix: str, opt: dict[str, np.ndarray], orig: dict[str, np.ndarray]
) -> None:
    np.savez_compressed(out_dir / f"{prefix}_optimizer_consistent_stats.npz", **opt)
    np.savez_compressed(out_dir / f"{prefix}_original_stats.npz", **orig)


def compute_stage_stats(
    result, config: dict[str, Any], sample_dir: Path
) -> dict[str, Any]:
    box = float(np.asarray(result.box_size)[0])
    radius = float(result.radius)
    stages = {
        "target": result.points_target,
        "init": result.points_init,
        "stage1": result.points_stage1,
        "stage2": result.points_stage2,
        "final": result.points_final,
    }
    stats: dict[str, Any] = {}
    for name, pts in stages.items():
        opt, opt_summary = optimizer_consistent_statistics(pts, radius, box, config)
        orig, orig_summary = original_statistics(pts, radius, None, box, config)
        save_stats_npz(sample_dir, name, opt, orig)
        stats[name] = {
            "points": pts,
            "optimizer": opt,
            "original": orig,
            "optimizer_summary": opt_summary,
            "original_summary": orig_summary,
        }
    return stats


def descriptor_errors(stats: dict[str, Any]) -> dict[str, dict[str, float]]:
    target = stats["target"]["optimizer"]
    out: dict[str, dict[str, float]] = {}
    for stage in ("stage1", "stage2", "final"):
        current = stats[stage]["optimizer"]
        out[stage] = {
            "RDF": mse(current["rdf_values"], target["rdf_values"]),
            "ARDF": mse(current["ardf_values"], target["ardf_values"]),
            "NND": mse(current["nn_values"], target["nn_values"]),
            "Voronoi": mse(
                current["voronoi_area_values"], target["voronoi_area_values"]
            ),
            "CV": float(abs(float(current["cv_value"]) - float(target["cv_value"]))),
            "Coord.": mse(current["coord_values"], target["coord_values"]),
            "Mean coord.": float(
                abs(
                    float(current["mean_coordination"])
                    - float(target["mean_coordination"])
                )
            ),
        }
    return out


def plot_configuration(
    stats: dict[str, Any], radius: float, box: float, fig_dir: Path
) -> None:
    fig, axes = setup_grid(
        2,
        2,
        figsize=(6.75, 6.75),
        left=0.155,
        right=0.945,
        bottom=0.080,
        top=0.955,
        wspace=0.30,
        hspace=0.48,
        square=True,
    )
    titles = ["Initial", "Stage 1", "Stage 2", "Reference RVE"]
    keys = ["init", "stage1", "stage2", "target"]
    flat_axes = np.asarray(axes).ravel()
    for ax, key, title in zip(flat_axes, keys, titles):
        draw_disks(ax, stats[key]["points"], radius, box, title)
    add_panel_labels(flat_axes, label_offset=(-36, 8))
    save_figure(fig, fig_dir / "configuration_evolution.png")


def plot_pair_statistics(
    stats: dict[str, Any],
    fig_dir: Path,
    *,
    fixed_image_scale: bool = False,
    fixed_random_scale: bool = False,
) -> None:
    target_o = stats["target"]["optimizer"]
    stage1_o = stats["stage1"]["optimizer"]
    final_o = stats["final"]["optimizer"]
    target_c = stats["target"]["original"]
    stage1_c = stats["stage1"]["original"]
    final_c = stats["final"]["original"]
    fig, axes = setup_grid(
        2,
        3,
        figsize=FOUR_PANEL_LAYOUT["figsize"],
        left=FOUR_PANEL_LAYOUT["left"],
        right=FOUR_PANEL_LAYOUT["right"],
        bottom=FOUR_PANEL_LAYOUT["bottom"],
        top=FOUR_PANEL_LAYOUT["top"],
        wspace=FOUR_PANEL_LAYOUT["wspace"],
        hspace=FOUR_PANEL_LAYOUT["hspace"],
        square=True,
    )
    third_column_shift = 0.039
    for ax in axes[:, 2]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + third_column_shift, pos.y0, pos.width, pos.height])

    search_radius_spec = PAIR_AXIS_SPECS[
        "search_radius_image"
        if fixed_image_scale
        else "search_radius_random"
        if fixed_random_scale
        else "search_radius_auto"
    ]
    soft_rdf_y_spec = PAIR_AXIS_SPECS[
        "soft_rdf_random" if fixed_random_scale else "soft_rdf_image"
    ]
    ardf_spec = (
        PAIR_COLORBAR_SPECS["image_ardf"]
        if fixed_image_scale
        else PAIR_COLORBAR_SPECS["random_ardf"]
        if fixed_random_scale
        else {}
    )
    soft_error_spec = (
        PAIR_COLORBAR_SPECS["image_soft_error"]
        if fixed_image_scale
        else PAIR_COLORBAR_SPECS["random_soft_error"]
        if fixed_random_scale
        else {}
    )
    classical_error_spec = (
        PAIR_COLORBAR_SPECS["image_classical_error"]
        if fixed_image_scale
        else PAIR_COLORBAR_SPECS["random_classical_error"]
        if fixed_random_scale
        else {}
    )

    def apply_search_radius_axis(ax, *, heatmap: bool = False) -> None:
        ax.set_xlabel(r"$r_s$", fontsize=AXIS_LABEL_FONTSIZE)
        apply_axis_spec(ax, x=search_radius_spec)

    def apply_theta_axis(ax) -> None:
        apply_axis_spec(ax, y=PAIR_AXIS_SPECS["theta"])

    scientific_colorbar_offsets = []
    manual_colorbar_offsets = []

    def add_scientific_colorbar(
        ax,
        image,
        ticks=None,
        *,
        scientific: bool = True,
        tick_scale: float | None = None,
        tick_decimals: int = 1,
        scientific_decimals: int | None = None,
        offset_text: str | None = None,
    ):
        cbar = add_external_colorbar(fig, ax, image)
        if ticks is not None:
            cbar.set_ticks(ticks)
        if tick_scale is not None:
            formatter = FuncFormatter(
                lambda value, _position: f"{value / tick_scale:.{tick_decimals}f}"
            )
        elif scientific:
            formatter_class = (
                FixedDecimalScalarFormatter
                if scientific_decimals is not None
                else ScalarFormatter
            )
            formatter_kwargs = (
                {"decimals": scientific_decimals}
                if scientific_decimals is not None
                else {}
            )
            formatter = formatter_class(useMathText=True, **formatter_kwargs)
            formatter.set_powerlimits((0, 0))
        else:
            formatter = FormatStrFormatter(f"%.{tick_decimals}f")
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=8.0, pad=4.0)
        offset_artist = cbar.ax.yaxis.get_offset_text()
        offset_artist.set_size(8.0)
        if tick_scale is not None and offset_text is not None:
            offset_artist.set_visible(False)
            manual_text = cbar.ax.text(
                0.0,
                1.0,
                offset_text,
                transform=cbar.ax.transAxes,
                ha="left",
                va="baseline",
                fontsize=8.0,
            )
            manual_colorbar_offsets.append((cbar, manual_text))
        elif scientific:
            scientific_colorbar_offsets.append((cbar, offset_artist))
        return cbar

    def align_manual_colorbar_offsets_to_scientific() -> None:
        if not scientific_colorbar_offsets or not manual_colorbar_offsets:
            return
        fig.canvas.draw()
        ref_cbar, ref_artist = scientific_colorbar_offsets[0]
        ref_anchor = ref_artist.get_transform().transform(ref_artist.get_position())
        ref_position = ref_cbar.ax.transAxes.inverted().transform(ref_anchor)
        for cbar, manual_text in manual_colorbar_offsets:
            manual_text.set_transform(cbar.ax.transAxes)
            manual_text.set_position(ref_position)
            manual_text.set_ha(ref_artist.get_ha())
            manual_text.set_va(ref_artist.get_va())

    axes[0, 0].set_prop_cycle(None)
    plot_mean_curve(
        axes[0, 0],
        target_o["rdf_grid"],
        target_o["rdf_values"],
        std_y=stats["target"].get("optimizer_std", {}).get("rdf_values"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[0, 0],
        stage1_o["rdf_grid"],
        stage1_o["rdf_values"],
        std_y=stats["stage1"].get("optimizer_std", {}).get("rdf_values"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[0, 0],
        final_o["rdf_grid"],
        final_o["rdf_values"],
        std_y=stats["final"].get("optimizer_std", {}).get("rdf_values"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    apply_search_radius_axis(axes[0, 0])
    apply_axis_spec(axes[0, 0], y=soft_rdf_y_spec)
    axes[0, 0].set_ylabel(r"Soft RDF")
    legend_upper_right(axes[0, 0])
    plot_mean_curve(
        axes[1, 0],
        target_c["rdf_grid"],
        target_c["rdf_values"],
        std_y=stats["target"].get("original_std", {}).get("rdf_values"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[1, 0],
        stage1_c["rdf_grid"],
        stage1_c["rdf_values"],
        std_y=stats["stage1"].get("original_std", {}).get("rdf_values"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[1, 0],
        final_c["rdf_grid"],
        final_c["rdf_values"],
        std_y=stats["final"].get("original_std", {}).get("rdf_values"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    apply_search_radius_axis(axes[1, 0])
    apply_axis_spec(axes[1, 0], y=PAIR_AXIS_SPECS["classical_rdf"])
    axes[1, 0].set_ylabel(r"$g(r)$")
    legend_upper_right(axes[1, 0])
    ardf_vmin = ardf_spec.get("vmin")
    ardf_vmax = ardf_spec.get(
        "vmax",
        max(
            float(np.max(target_o["ardf_values"])),
            float(np.max(final_o["ardf_values"])),
            1.0e-12,
        ),
    )
    ardf_ticks = ardf_spec.get("ticks")
    ardf_scientific_decimals = ardf_spec.get("scientific_decimals")
    sample_stats = stats.get("_sample_stats", [])
    if sample_stats:
        diff = np.mean(
            [
                np.abs(
                    sample["final"]["optimizer"]["ardf_values"]
                    - sample["target"]["optimizer"]["ardf_values"]
                )
                for sample in sample_stats
            ],
            axis=0,
        )
    else:
        diff = np.abs(final_o["ardf_values"] - target_o["ardf_values"])
    extent_soft = [
        float(target_o["ardf_r_grid"][0]),
        float(target_o["ardf_r_grid"][-1]),
        0.0,
        np.pi,
    ]
    im0 = axes[0, 1].imshow(
        target_o["ardf_values"].T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=ardf_vmin,
        vmax=ardf_vmax,
        extent=extent_soft,
    )
    apply_search_radius_axis(axes[0, 1], heatmap=True)
    apply_theta_axis(axes[0, 1])
    axes[0, 1].set_ylabel(r"$\theta$")
    axes[0, 1].set_title("Reference ARDF", fontsize=PANEL_TITLE_FONTSIZE)
    add_scientific_colorbar(
        axes[0, 1], im0, ticks=ardf_ticks, scientific_decimals=ardf_scientific_decimals
    )
    im1 = axes[0, 2].imshow(
        final_o["ardf_values"].T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=ardf_vmin,
        vmax=ardf_vmax,
        extent=extent_soft,
    )
    apply_search_radius_axis(axes[0, 2], heatmap=True)
    apply_theta_axis(axes[0, 2])
    axes[0, 2].set_ylabel(r"$\theta$")
    axes[0, 2].set_title("Reconstruction ARDF", fontsize=PANEL_TITLE_FONTSIZE)
    add_scientific_colorbar(
        axes[0, 2], im1, ticks=ardf_ticks, scientific_decimals=ardf_scientific_decimals
    )
    lim = soft_error_spec.get("vmax", max(float(np.max(diff)), 1.0e-12))
    soft_error_ticks = soft_error_spec.get("ticks")
    im2 = axes[1, 1].imshow(
        diff.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=lim,
        extent=extent_soft,
    )
    apply_search_radius_axis(axes[1, 1], heatmap=True)
    apply_theta_axis(axes[1, 1])
    axes[1, 1].set_ylabel(r"$\theta$")
    axes[1, 1].set_title("Soft ARDF error", fontsize=PANEL_TITLE_FONTSIZE)
    soft_tick_scale = soft_error_spec.get("tick_scale")
    soft_tick_decimals = soft_error_spec.get("tick_decimals", 1)
    soft_offset_text = soft_error_spec.get("offset_text")
    add_scientific_colorbar(
        axes[1, 1],
        im2,
        ticks=soft_error_ticks,
        tick_scale=soft_tick_scale,
        tick_decimals=soft_tick_decimals,
        offset_text=soft_offset_text,
    )
    class_abs_diff = np.abs(final_c["ardf_values"] - target_c["ardf_values"])
    if sample_stats:
        if fixed_random_scale:
            class_diff = np.mean(
                [
                    np.abs(
                        sample["final"]["original"]["ardf_values"]
                        - sample["target"]["original"]["ardf_values"]
                    )
                    / np.maximum(
                        np.maximum(
                            np.abs(sample["final"]["original"]["ardf_values"]),
                            np.abs(sample["target"]["original"]["ardf_values"]),
                        ),
                        1.0e-12,
                    )
                    for sample in sample_stats
                ],
                axis=0,
            )
        else:
            class_diff = np.mean(
                [
                    np.abs(
                        sample["final"]["original"]["ardf_values"]
                        - sample["target"]["original"]["ardf_values"]
                    )
                    for sample in sample_stats
                ],
                axis=0,
            )
    else:
        if fixed_random_scale:
            class_denom = np.maximum(
                np.maximum(
                    np.abs(final_c["ardf_values"]), np.abs(target_c["ardf_values"])
                ),
                1.0e-12,
            )
            class_diff = class_abs_diff / class_denom
        else:
            class_diff = class_abs_diff
    lim_c = classical_error_spec.get("vmax", max(float(np.max(class_diff)), 1.0e-12))
    classical_error_ticks = classical_error_spec.get("ticks")
    extent_classic = [
        float(target_c["ardf_r_grid"][0]),
        float(target_c["ardf_r_grid"][-1]),
        0.0,
        np.pi,
    ]
    im3 = axes[1, 2].imshow(
        class_diff.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=lim_c,
        extent=extent_classic,
    )
    apply_search_radius_axis(axes[1, 2], heatmap=True)
    apply_theta_axis(axes[1, 2])
    axes[1, 2].set_ylabel(r"$\theta$")
    axes[1, 2].set_title("Classical ARDF error", fontsize=PANEL_TITLE_FONTSIZE)
    classical_tick_scale = classical_error_spec.get("tick_scale")
    classical_tick_decimals = classical_error_spec.get("tick_decimals", 1)
    classical_offset_text = classical_error_spec.get("offset_text")
    add_scientific_colorbar(
        axes[1, 2],
        im3,
        ticks=classical_error_ticks,
        scientific=not fixed_random_scale,
        tick_scale=classical_tick_scale,
        tick_decimals=classical_tick_decimals,
        offset_text=classical_offset_text,
    )
    align_manual_colorbar_offsets_to_scientific()
    add_panel_labels(axes)
    save_figure(fig, fig_dir / "pair_statistics.png")


def plot_local_statistics(stats: dict[str, Any], fig_dir: Path) -> None:
    target_o = stats["target"]["optimizer"]
    stage1_o = stats["stage1"]["optimizer"]
    final_o = stats["final"]["optimizer"]
    target_c = stats["target"]["original"]
    stage1_c = stats["stage1"]["original"]
    final_c = stats["final"]["original"]
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
    plot_mean_curve(
        axes[0, 0],
        target_o["nn_grid"],
        target_o["nn_values"],
        std_y=stats["target"].get("optimizer_std", {}).get("nn_values"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[0, 0],
        stage1_o["nn_grid"],
        stage1_o["nn_values"],
        std_y=stats["stage1"].get("optimizer_std", {}).get("nn_values"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[0, 0],
        final_o["nn_grid"],
        final_o["nn_values"],
        std_y=stats["final"].get("optimizer_std", {}).get("nn_values"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    axes[0, 0].set_xlabel(r"$d_{\rm nn}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 0].set_ylabel("Soft NND")
    apply_axis_spec(
        axes[0, 0], x=LOCAL_AXIS_SPECS["soft_nnd_x"], y=LOCAL_AXIS_SPECS["soft_nnd_y"]
    )
    plot_mean_curve(
        axes[0, 1],
        target_c["nn_grid"],
        target_c["nn_probability"],
        std_y=stats["target"].get("original_std", {}).get("nn_probability"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[0, 1],
        stage1_c["nn_grid"],
        stage1_c["nn_probability"],
        std_y=stats["stage1"].get("original_std", {}).get("nn_probability"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[0, 1],
        final_c["nn_grid"],
        final_c["nn_probability"],
        std_y=stats["final"].get("original_std", {}).get("nn_probability"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    axes[0, 1].set_xlabel(r"$d_{\rm nn}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 1].set_ylabel("Probability")
    apply_axis_spec(
        axes[0, 1],
        x=LOCAL_AXIS_SPECS["classical_nnd_x"],
        y=LOCAL_AXIS_SPECS["classical_nnd_y"],
    )
    plot_mean_curve(
        axes[1, 0],
        target_o["voronoi_area_grid"],
        target_o["voronoi_area_values"],
        std_y=stats["target"].get("optimizer_std", {}).get("voronoi_area_values"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[1, 0],
        stage1_o["voronoi_area_grid"],
        stage1_o["voronoi_area_values"],
        std_y=stats["stage1"].get("optimizer_std", {}).get("voronoi_area_values"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[1, 0],
        final_o["voronoi_area_grid"],
        final_o["voronoi_area_values"],
        std_y=stats["final"].get("optimizer_std", {}).get("voronoi_area_values"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    axes[1, 0].set_xlabel(r"$A_{\rm vor}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 0].set_ylabel("Soft density")
    apply_axis_spec(
        axes[1, 0],
        x=LOCAL_AXIS_SPECS["soft_voronoi_x"],
        y=LOCAL_AXIS_SPECS["soft_voronoi_y"],
    )
    plot_mean_curve(
        axes[1, 1],
        target_c["voronoi_area_grid"],
        target_c["voronoi_area_probability"],
        std_y=stats["target"].get("original_std", {}).get("voronoi_area_probability"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[1, 1],
        stage1_c["voronoi_area_grid"],
        stage1_c["voronoi_area_probability"],
        std_y=stats["stage1"].get("original_std", {}).get("voronoi_area_probability"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[1, 1],
        final_c["voronoi_area_grid"],
        final_c["voronoi_area_probability"],
        std_y=stats["final"].get("original_std", {}).get("voronoi_area_probability"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    axes[1, 1].set_xlabel(r"$A_{\rm vor}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 1].set_ylabel("Probability")
    apply_axis_spec(
        axes[1, 1],
        x=LOCAL_AXIS_SPECS["classical_voronoi_x"],
        y=LOCAL_AXIS_SPECS["classical_voronoi_y"],
    )
    for ax in axes.flat:
        legend_upper_right(ax)
    add_panel_labels(axes)
    save_figure(fig, fig_dir / "local_statistics.png")


def plot_coordination(stats: dict[str, Any], fig_dir: Path) -> None:
    target_o = stats["target"]["optimizer"]
    stage1_o = stats["stage1"]["optimizer"]
    final_o = stats["final"]["optimizer"]
    target_c = stats["target"]["original"]
    stage1_c = stats["stage1"]["original"]
    final_c = stats["final"]["original"]
    fig, axes = setup_grid(
        1,
        2,
        figsize=(6.9, 2.8),
        left=0.095,
        right=0.965,
        bottom=0.18,
        top=0.93,
        wspace=0.15,
        hspace=0.275,
        square=True,
    )
    plot_mean_curve(
        axes[0],
        target_o["coord_grid"],
        target_o["coord_values"],
        std_y=stats["target"].get("optimizer_std", {}).get("coord_values"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[0],
        stage1_o["coord_grid"],
        stage1_o["coord_values"],
        std_y=stats["stage1"].get("optimizer_std", {}).get("coord_values"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[0],
        final_o["coord_grid"],
        final_o["coord_values"],
        std_y=stats["final"].get("optimizer_std", {}).get("coord_values"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    axes[0].set_xlabel(r"$z$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel("Soft coordination")
    apply_axis_spec(axes[0], x=COORD_AXIS_SPECS["soft_x"], y=COORD_AXIS_SPECS["soft_y"])
    plot_mean_curve(
        axes[1],
        target_c["coord_grid"],
        target_c["coord_probability"],
        std_y=stats["target"].get("original_std", {}).get("coord_probability"),
        label="Reference",
        style=apply_standard_line_style(0),
    )
    plot_mean_curve(
        axes[1],
        stage1_c["coord_grid"],
        stage1_c["coord_probability"],
        std_y=stats["stage1"].get("original_std", {}).get("coord_probability"),
        label="Overlap-removal",
        style=apply_standard_line_style(1),
    )
    plot_mean_curve(
        axes[1],
        final_c["coord_grid"],
        final_c["coord_probability"],
        std_y=stats["final"].get("original_std", {}).get("coord_probability"),
        label="Reconstruction",
        style=apply_standard_line_style(2),
    )
    axes[1].set_xlabel(r"$z$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel("Probability")
    apply_axis_spec(
        axes[1], x=COORD_AXIS_SPECS["classical_x"], y=COORD_AXIS_SPECS["classical_y"]
    )
    for ax in axes:
        legend_upper_right(ax)
    add_panel_labels(axes)
    save_figure(fig, fig_dir / "coordination_statistics.png")


def plot_errors(
    errors: dict[str, dict[str, float]],
    fig_dir: Path,
    errors_std: dict[str, dict[str, float]] | None = None,
) -> None:
    labels = list(next(iter(errors.values())).keys())
    short_labels = [DESCRIPTOR_LABELS.get(label, label) for label in labels]
    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(4.7, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.95)
    style_axis(ax)
    stage_labels = {"stage1": "Overlap-removal", "stage2": "Stage 2"}
    for i, stage in enumerate(("stage1", "stage2")):
        vals = [errors[stage][label] for label in labels]
        yerr = None
        if errors_std is not None:
            yerr = [errors_std[stage][label] for label in labels]
        color = [COLORS["gray"], COLORS["blue"]][i]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            yerr=yerr,
            capsize=3,
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
    apply_axis_spec(
        ax, x=ERROR_AXIS_SPECS["descriptor_x"], y=ERROR_AXIS_SPECS["descriptor_y"]
    )
    legend_upper_left(ax)
    add_panel_labels([ax])
    save_figure(fig, fig_dir / "error_reduction.png")


def plot_convergence(result, fig_dir: Path) -> None:
    hist = result.history
    fig, axes = setup_grid(
        1,
        2,
        figsize=(6.9, 2.8),
        left=0.10,
        right=0.965,
        bottom=0.18,
        top=0.94,
        wspace=0.30,
        hspace=0.275,
        square=True,
    )
    if hist:
        it = [row["iteration"] for row in hist]
        axes[0].plot(
            it,
            [row["loss_total"] for row in hist],
            label="Total",
            **apply_standard_line_style(0),
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
            axes[0].plot(
                it,
                [row[key] for row in hist],
                label=label,
                **apply_standard_line_style(idx),
            )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Stage-2 iteration")
        axes[0].set_ylabel("Loss")
        apply_axis_spec(
            axes[0],
            x=CONVERGENCE_AXIS_SPECS["loss_x"],
            y=CONVERGENCE_AXIS_SPECS["loss_y"],
        )
        axes[0].legend(
            frameon=False,
            loc="upper right",
            ncol=3,
            borderaxespad=0.2,
            columnspacing=0.8,
            handlelength=1.4,
            handletextpad=0.4,
        )
    potentials = [
        result.stage_metrics["stage1"]["overlap_potential"],
        result.stage_metrics["stage2"].get(
            "final_overlap_loss", result.final_terms.get("loss_overlap", 0.0)
        ),
    ]
    y_min = (
        CONVERGENCE_AXIS_SPECS["overlap_y"].limits[0]
        if CONVERGENCE_AXIS_SPECS["overlap_y"].limits is not None
        else 1.0e-16
    )
    plot_potentials = [max(value, y_min) for value in potentials]
    x = [0.35, 0.65]
    axes[1].bar(
        x,
        [value - y_min for value in plot_potentials],
        bottom=y_min,
        width=0.14,
        color=[COLORS["blue"], COLORS["orange"]],
    )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Stage 1", "Stage 2"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Overlap potential")
    apply_axis_spec(axes[1], y=CONVERGENCE_AXIS_SPECS["overlap_y"])
    add_panel_labels(axes)
    save_figure(fig, fig_dir / "convergence.png")
