"""Generate Figure 8 for the image-derived reference."""

from __future__ import annotations

import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))

from _figure_common import (  # noqa: E402
    compute_case_average_stage_data,
    copy_figure_files,
    remove_figure_files,
)
from plotting.manuscript_plot_style import AxisSpec  # noqa: E402
from rve_study import rve_study as study  # noqa: E402


SOURCE_NAME = "local_statistics"
FIGURE_NAME = "Figure-8"
CASE = "image_reference"
AXIS_SPECS = {
    "soft_nnd_x": AxisSpec(
        limits=(0.00, 1.20), tick_start=0.00, tick_stop=1.20, tick_step=0.30, decimals=1
    ),
    "soft_nnd_y": AxisSpec(
        limits=(0.00, 0.16), tick_start=0.00, tick_stop=0.16, tick_step=0.04, decimals=2
    ),
    "classical_nnd_x": AxisSpec(
        limits=(0.00, 1.20), tick_start=0.00, tick_stop=1.20, tick_step=0.30, decimals=1
    ),
    "classical_nnd_y": AxisSpec(
        limits=(0.00, 1.20), tick_start=0.00, tick_stop=1.20, tick_step=0.30, decimals=2
    ),
    "soft_voronoi_x": AxisSpec(
        limits=(0.00, 0.80), tick_start=0.00, tick_stop=0.80, tick_step=0.20, decimals=1
    ),
    "soft_voronoi_y": AxisSpec(
        limits=(0.00, 0.12), tick_start=0.00, tick_stop=0.12, tick_step=0.03, decimals=2
    ),
    "classical_voronoi_x": AxisSpec(
        limits=(0.00, 0.80), tick_start=0.00, tick_stop=0.80, tick_step=0.20, decimals=1
    ),
    "classical_voronoi_y": AxisSpec(
        limits=(0.00, 0.32), tick_start=0.00, tick_stop=0.32, tick_step=0.08, decimals=2
    ),
}


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    _, _, _, stats, _, _ = compute_case_average_stage_data(CASE, sample_count=10)
    study.LOCAL_AXIS_SPECS.update(AXIS_SPECS)
    study.plot_local_statistics(stats, output_dir)
    copy_figure_files(output_dir, SOURCE_NAME, output_dir, FIGURE_NAME)
    remove_figure_files(output_dir, SOURCE_NAME)


if __name__ == "__main__":
    main()
