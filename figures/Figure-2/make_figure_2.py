"""Generate Figure 2 from the matched-reference comparison data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
DATA_DIR = FIGURE_DIR.parent / "data" / "baseline_comparison"


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(FIGURE_DIR / "scripts" / "plot_benchmark.py"),
            "--runs-csv",
            str(DATA_DIR / "runs.csv"),
            "--out",
            str(FIGURE_DIR / "Figure-2.png"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
