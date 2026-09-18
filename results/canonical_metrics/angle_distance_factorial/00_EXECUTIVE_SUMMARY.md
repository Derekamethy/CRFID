# Executive summary

## Outcome: `ANGLE_ASSOCIATED_DEGRADATION`

The inherited 60-run fixed-position benchmark is not fully paired across P1-P4:
492 train/validation assignment groups differ, although outer test folds match.
The released metrics and confusion matrices were reconstructed exactly, then a
preregistered Case B rerun trained exactly 60 synchronized C1 runs with no tuning.
Factorial inference uses only that synchronized rerun.

Primary paired condition-block Accuracy (63 physical blocks):

- angle, 45 degrees minus 0 degrees: `-0.111111 [-0.160317, -0.063492]`;
- distance, 150 mm minus 50 mm: `-0.050794 [-0.106349, 0.007937]`;
- interaction: `0.114286 [0.000000, 0.228571]`.

Angle-associated degradation was confirmed.
The distance effect was not confirmed,
and the interaction was not confirmed.
These are performance associations, not causal RF claims.

The positive interaction point estimate is not robust to paired seed resampling;
it is reported as secondary uncertainty rather than a confirmed interaction.

The primary interval is a 10,000-replicate paired, TagID-stratified physical-block
bootstrap with seeds fixed. A paired block-plus-seed sensitivity analysis is
reported separately. All four inherited learning-validity checks passed, but
learning validity is not held-condition generalisation.

The public result retains compact condition metadata, paired aggregate evidence, and bounded interpretation. Raw rows, checkpoints, logits, embeddings, and prediction arrays are not included here.
