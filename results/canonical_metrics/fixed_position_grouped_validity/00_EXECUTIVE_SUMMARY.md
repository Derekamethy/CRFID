# Fixed-position grouped-validity executive summary

## Outcome

The preregistered benchmark **ran fully**: 4 positions x 3 outer folds x 5
seeds = **60 primary runs**. All ten pre-training structure, leakage,
compatibility, isolation, and determinism gates passed. P1, P2, P3, and P4 each
passed the independent tiny-subset learning-validity check.

The evidence is a **mixed position-specific pattern (interpretation D)**. P1
and P3 show condition-general TagID signal above the balanced 1/7 baseline.
P2 and P4 remain weak and uncertain: their block-bootstrap intervals overlap
the baseline. P4 is therefore **not uniquely difficult**; P2 is at least as
difficult under this protocol.

| Position | Learning validity | Train Accuracy mean | Validation Accuracy mean | Held row Accuracy mean ± population SD | 95% block-cluster CI | Held row Macro-F1 mean ± population SD | pooled-block Macro-F1 point [95% CI] | Held block Accuracy mean ± population SD | Train-to-held gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | PASS | 0.6935 | 0.2874 | 0.2688 ± 0.0775 | [0.2147, 0.3257] | 0.1998 ± 0.0907 | 0.2348 [0.1788, 0.2846] | 0.2667 ± 0.0793 | 0.4247 |
| P2 | PASS | 0.6236 | 0.2270 | 0.1499 ± 0.0564 | [0.1081, 0.1948] | 0.0885 ± 0.0607 | 0.1194 [0.0771, 0.1581] | 0.1492 ± 0.0573 | 0.4737 |
| P3 | PASS | 0.7733 | 0.2658 | 0.2389 ± 0.0799 | [0.1902, 0.2927] | 0.1697 ± 0.0910 | 0.2164 [0.1631, 0.2649] | 0.2444 ± 0.0849 | 0.5344 |
| P4 | PASS | 0.4749 | 0.1748 | 0.1711 ± 0.0573 | [0.1300, 0.2154] | 0.0861 ± 0.0430 | 0.1322 [0.0929, 0.1699] | 0.1714 ± 0.0597 | 0.3037 |

Uniform chance, empirical row-majority, and condition-block-majority baselines
are all 1/7 = 0.142857 because every evaluation fold is class-balanced.

## Learning-validity checks

The canonical CNN fitted the fixed 56-row true-label diagnostic subset at
every position without altering primary hyperparameters: P1 reached Accuracy
0.9643 / Macro-F1 0.9641 in 37 steps; P2 0.9821 / 0.9821 in 38; P3 0.9821 /
0.9821 in 33; and P4 0.9643 / 0.9637 in 50. No position is classified
`LEARNING_VALIDITY_FAILURE`.

## What the benchmark supports

- Within-position unseen-condition TagID generalisation exists for P1 and P3
  in this dataset and protocol: both Accuracy and pooled condition-block
  Macro-F1 intervals lie above 1/7.
- There is no clear above-baseline evidence for P2 or P4; both Accuracy and
  pooled-block Macro-F1 intervals overlap 1/7.
- P4 is not a unique failure mode. P2 has lower mean held Accuracy than P4,
  while their held Macro-F1 means are nearly equal.
- Large and variable train-to-held gaps show that fitting fixed-position
  training blocks does not reliably transfer to unseen ER/surface blocks.

## Required caveats

The 95% intervals resample the 63 observed condition blocks within TagID and
pair the five seed predictions; they do not treat the 50 repeats as independent.
For nonlinear Macro-F1, the 15-run descriptive mean and the preregistered
pooled-63-block bootstrap point are different estimands, so both are displayed.
The physical frequency axis remains unresolved; the canonical ordered 281-point
carrier is used. The study covers nine ER/surface combinations per TagID at
each position and does not establish performance on new datasets, new tags, or
new measurement positions.

Raw data, processed arrays, checkpoints, embeddings, logits, and full
predictions are not distributed in the public result package. The public copy
retains compact aggregate evidence and the grouped-validity interpretation.
