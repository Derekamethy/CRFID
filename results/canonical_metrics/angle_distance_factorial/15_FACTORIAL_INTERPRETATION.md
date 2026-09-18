# Factorial interpretation

## Classification: `ANGLE_ASSOCIATED_DEGRADATION`

The inherited 60 fixed-position runs are not fully paired: their outer test folds
are synchronized, but 492 train/validation assignment groups differ across
positions. Their frozen checkpoints exactly reproduce the released benchmark,
but those runs are not used for factorial inference. Case B therefore ran exactly
60 synchronized C1 primary runs with the same outer folds and seeds, no tuning,
and position-independent validation assignment. The new frozen checkpoints also
exactly reproduce their compact held row/block metrics and confusion matrices.

The primary endpoint is paired condition-block Accuracy over 63 physical cells,
with all five seeds retained inside each paired block bootstrap unit.

| Primary contrast | Estimate and paired 95% interval |
|---|---:|
| 45 degrees minus 0 degrees | -0.111111 [-0.160317, -0.063492] |
| 150 mm minus 50 mm | -0.050794 [-0.106349, 0.007937] |
| Angle x distance interaction | 0.114286 [0.000000, 0.228571] |

- Angle-associated degradation confirmed: **yes**.
- Distance effect confirmed: **no**.
- Angle-distance interaction confirmed: **no**.

The corresponding pooled condition-block Macro-F1 contrasts are angle
`-0.109330 [-0.154894, -0.061699]`, distance
`-0.052759 [-0.105286, 0.001141]`, and interaction
`0.116263 [0.007935, 0.223512]`. Macro-F1 is
computed after pooling all 63 aligned blocks within position and seed, then
averaging the five seed-specific values. It is not a per-block metric, and its
pooled estimand is separate from the inherited descriptive 15-run mean.

The fixed-seed Macro-F1 interaction interval excludes zero, but the primary
Accuracy interval touches zero and both paired block-plus-seed interaction
sensitivity intervals cross zero. The interaction is therefore not confirmed.

The angle statement is associational: held-condition classification performance
is lower at 45 degrees in this benchmark. It does not identify a causal RF
mechanism. TagID, ER, surface, seed, fold, and simple-effect summaries are
secondary or exploratory and were not searched for a preferred narrative.

Basic learning validity also remains separate: all four inherited tiny-set tests
passed (`learning_validity_passed=true`), but that fact
does not establish held-condition transfer or a factor effect.
