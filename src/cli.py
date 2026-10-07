"""Command-line interface for pure mathematical RVE optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from config import load_config
from optimizer import optimize_rve_2d, save_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_2d.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    project = config.setdefault("project", {})
    if args.output_dir is not None:
        project["output_dir"] = args.output_dir
    if args.num_samples is not None:
        project["num_samples"] = args.num_samples

    output_dir = Path(project.get("output_dir", "outputs/default_2d"))
    num_samples = int(project.get("num_samples", 1))
    rows = []
    for index in range(num_samples):
        result = optimize_rve_2d(config, seed_offset=index)
        path = save_result(result, output_dir, index=index)
        row = {
            "index": index,
            "file": str(path),
            "runtime_seconds": result.runtime_seconds,
            "stage_metrics": result.stage_metrics,
            "final_terms": result.final_terms,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    (output_dir / "run_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
