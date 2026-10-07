"""Make Figure 7 pair statistics for the numerical reference."""

from __future__ import annotations

import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))

from _figure_common import (
    compute_case_average_stage_data,
    copy_figure_files,
    remove_figure_files,
)  # noqa: E402
from rve_study.rve_study import plot_pair_statistics  # noqa: E402


SOURCE_NAME = "pair_statistics"
FIGURE_NAME = "Figure-7"
CASE = "random_reference"


def main() -> None:
    local_dir = Path(__file__).resolve().parent
    _, _, _, stats, _, _ = compute_case_average_stage_data(CASE, sample_count=10)
    plot_pair_statistics(
        stats, local_dir, fixed_image_scale=True, fixed_random_scale=False
    )
    copy_figure_files(local_dir, SOURCE_NAME, local_dir, FIGURE_NAME)
    remove_figure_files(local_dir, SOURCE_NAME)


if __name__ == "__main__":
    main()
