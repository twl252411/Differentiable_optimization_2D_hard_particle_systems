# Figure 3: initialization sensitivity

Run `python figures/Figure-3/make_initialization_sensitivity_20.py` from the
repository root. The script reads
`manuscript_data/aggregates/initialization_sensitivity_20_summary.csv` and
`manuscript_data/aggregates/initialization_acceptance_audit.csv` and writes
`Figure-3.png`, `Figure-3.pdf`, and `Figure-3.svg` beside the script.

Each bar shows the arithmetic mean over 20 attempts; the error bar is one
sample standard deviation. The study uses the repository-wide Stage-2 target
loss of `1e-4`. Exact stopping and geometry counts are read from
`initialization_acceptance_audit.csv` and annotated in panel (d).
All 160 attempts were accepted, with 20/20 accepted in every
target/initialization group.

The axis limits, tick labels, four-panel layout, and colors follow the
manuscript figure style. Acceptance is annotated in panel (d).
