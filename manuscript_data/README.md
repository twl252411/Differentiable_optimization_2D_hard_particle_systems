# Manuscript data map

This directory contains the final manuscript-level aggregates and metadata
needed to trace the reported analysis.
Temporary run files and proprietary Abaqus databases are excluded.

`MANUSCRIPT_MAPPING.csv` gives a row-by-row map from manuscript claims and
figures to the corresponding public source, while identifying the omitted FE
source data.

`FIGURE_REPRODUCIBILITY.csv` audits every main-text, appendix, and supplementary
figure separately. It identifies the final stored data and plotting program
for each figure. Author-supplied PDFs are marked separately.

The corresponding plotting programs and 10-run statistical figure data are
under `figures/`. Run `python figures/make_all_figures.py` from the repository
root to regenerate Figures 2--11. Figures 1, C.1, D.1, E.1, and E.2 are
included as author-supplied PDFs. Elastic and damage source data are excluded.

## Manuscript aggregates

The `aggregates/` directory contains the processed values used for, or reported
alongside, the manuscript analyses:

| File | Scope |
| --- | --- |
| `initialization_sensitivity_20_summary.csv` | 20-run summary for four initializations: regular, fixed-count Poisson, scrambled Sobol, and periodic Thomas cluster |
| `initialization_acceptance_audit.csv` | stopping-rule and full-precision geometry acceptance counts for the same 160 runs |
| `size_scaling_reported.csv` | final reported values from the 120-run, target-loss-`1e-4` size-scaling study, including success rate, sample SD, and exact geometry checks |

The size-scaling file is derived from 10 attempts per row and separates the
weighted soft objective from the conventional descriptor discrepancy. All 120
runs reached the `1e-4` Stage-2 target and passed the full-precision periodic
non-overlap check.

Runtime covers Stage 1, Stage 2, and the terminal projection. Reported
iteration counts sum the Stage-1 and Stage-2 optimizer iterations and exclude
projection sweeps. The final soft objective is the weighted differentiable
objective. `minimum_normalized_gap` is defined as
`(s_ij - 2 r_h) / (2 r_h)`, using the minimum-image center distance and the
hard-core radius. `maximum_penetration` is zero for an exact non-overlapping
terminal state. A run is accepted only when its Stage-2 stopping and terminal
geometry checks both pass.

Figure 3 means and sample standard deviations include all 20 attempts in each
target/initialization group. Exact stopping, geometry, and joint acceptance
counts are in `aggregates/initialization_acceptance_audit.csv`. The final study
accepted all 160 attempts, with 20/20 accepted in every group.

The matched benchmark data contain `runs.csv` and `summary_by_method.csv`,
covering five paired seeds for each combination of five methods and two
targets. FE workflow code is available, but elastic and damage source data are
intentionally omitted.
