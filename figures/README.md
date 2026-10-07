# Manuscript figures

The figure programs and their input data are listed below. Paths are relative to
the repository root. The batch command regenerates Figures 2--11. Figures 1,
C.1, D.1, E.1, and E.2 are author-supplied original PDFs.

The differentiable results use a Stage-2 target loss of `1e-4`.
Only the final data read by the plotting programs are included.

Run the batch command with:

```powershell
python figures/make_all_figures.py
```

| Manuscript asset | Output | Program | Data |
| --- | --- | --- | --- |
| Final-configuration schematic | `Figure-1` | Author PDF and Origin project | No source data required |
| Matched-reference comparison | `Figure-2` | `figures/Figure-2/make_figure_2.py` | `figures/data/baseline_comparison/` |
| Initialization sensitivity | `Figure-3` | `figures/Figure-3/make_initialization_sensitivity_20.py` | `manuscript_data/aggregates/initialization_sensitivity_20_summary.csv` and `initialization_acceptance_audit.csv` |
| Convergence | `Figure-4` | `figures/Figure-4/make_convergence.py` | `figures/data/reference_results/` |
| Configuration evolution | `Figure-5` | `figures/Figure-5/make_configuration_evolution.py` | `figures/data/reference_results/` and `image_detection/` |
| Pair statistics, image reference | `Figure-6` | `figures/Figure-6/make_pair_statistics.py` | `figures/data/image_reference/` |
| Pair statistics, numerical reference | `Figure-7` | `figures/Figure-7/make_pair_statistics.py` | `figures/data/random_reference/` |
| Local statistics, image reference | `Figure-8` | `figures/Figure-8/make_local_statistics.py` | `figures/data/image_reference/` |
| Local statistics, numerical reference | `Figure-9` | `figures/Figure-9/make_local_statistics.py` | `figures/data/random_reference/` |
| Coordination statistics | `Figure-10` | `figures/Figure-10/make_coordination_statistics.py` | same 10-run descriptor archive |
| Descriptor error reduction | `Figure-11` | `figures/Figure-11/make_error_reduction.py` | same 10-run descriptor archive |
| Supplementary descriptor validation | `Figure-C1` | Author PDF | No source data required |
| Image-reference detection | `Figure-D1` | Author PDF | No source data required |
| Elastic illustration | `Figure-E1` | Author PDF | Elastic source data excluded |
| Nonlinear damage curves | `Figure-E2` | Author PDF | Damage source data excluded |

Elastic and damage source data, including Abaqus outputs, are excluded. The
author-supplied PDFs are excluded from `make_all_figures.py`.
Figure 3 uses mean \(\pm\) one sample standard deviation over all 20 attempts
per group. Panel (d) reports the joint acceptance counts stored in
`manuscript_data/aggregates/initialization_acceptance_audit.csv`.
