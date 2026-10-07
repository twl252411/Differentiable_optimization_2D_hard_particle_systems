# Matched-reference manuscript benchmark

This directory contains the data and plotting tools for the
five-method comparison. The image-based and numerical targets share
`N = 349`, `L = 10`, nominal radius `0.2168117479`, hard-core radius
`0.2222320415`, and identical descriptor, weight, initialization, and stopping
settings. Only the target center arrangement differs.

`../data/baseline_comparison/` contains five paired seeds for each of
five methods and two targets. `runs.csv` contains the final plotted values and
`summary_by_method.csv` contains the corresponding method-level summary.

Regenerate the four-panel figure with:

```powershell
python figures/Figure-2/scripts/plot_benchmark.py `
  --runs-csv figures/data/baseline_comparison/runs.csv `
  --summary-csv figures/data/baseline_comparison/summary_by_method.csv `
  --out figures/Figure-2/Figure-2.png
```

The panels report runtime, conventional descriptor discrepancy `E_conv`, the
full-precision minimum normalized gap, and method-specific update counts. The
last quantity remains descriptive because update costs differ across methods.
All terminal feasibility projections belong to Stage 2; no Stage 3 is used.
