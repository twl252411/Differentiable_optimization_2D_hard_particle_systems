# Figure data

This directory contains the numerical data used by the manuscript
figure programs. Current differentiable runs use a Stage-2 target loss of
`1e-4`.

- `baseline_comparison/`: final Figure 2 plotted records and aggregate table.
- `reference_results/`: representative optimization histories used by Figures 4 and 5.
- `image_reference/`: ten-run image-reference descriptor statistics.
- `random_reference/`: ten-run numerical-reference descriptor statistics.

Figure 2's `conv_error_average` is the arithmetic mean of seven conventional
post-processed relative errors. It is distinct from the weighted
differentiable `soft_loss_total`. Its normalized gap is
`(s_ij - 2 r_h) / (2 r_h)`. The stored values are the final values used in
Figure 2.
