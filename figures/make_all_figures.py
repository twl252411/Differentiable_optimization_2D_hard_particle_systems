"""Regenerate manuscript Figures 2–11 from the supplied data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PROGRAMS = (
    "figures/Figure-2/make_figure_2.py",
    "figures/Figure-3/make_initialization_sensitivity_20.py",
    "figures/Figure-4/make_convergence.py",
    "figures/Figure-5/make_configuration_evolution.py",
    "figures/Figure-6/make_pair_statistics.py",
    "figures/Figure-7/make_pair_statistics.py",
    "figures/Figure-8/make_local_statistics.py",
    "figures/Figure-9/make_local_statistics.py",
    "figures/Figure-10/make_coordination_statistics.py",
    "figures/Figure-11/make_error_reduction.py",
)


def main() -> None:
    for program in FIGURE_PROGRAMS:
        print(f"Running {program}", flush=True)
        subprocess.run([sys.executable, program], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
