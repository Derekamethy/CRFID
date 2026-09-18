# Preregistration: P4 Trainable Linear-Readout Calibration Curve v1

The normative specification is
`configs/p4_trainable_linear_readout/preregistration.json`. Its SHA-256 is sealed in
the adjacent `preregistration.sha256` file before any new P4 readout is fitted.

## Question and hypotheses

The experiment asks whether the negative larger P4 calibration curve was principally
caused by the single-prototype adapter or by a frozen C1 representation that does not
transfer linearly across held P4 physical-condition blocks.

- H1 adapter bottleneck: a discriminatively trained linear head reliably and
  practically outperforms the matched prototype curve and strict-DG anchor.
- H2 representation bottleneck: the linear head provides no reliable recovery under
  the same supports and held-condition queries.

## Frozen design

The parent commit, P4 files, preprocessing, five encoders, three outer folds, twenty
support seeds, exact support rows, budgets, queries, metrics, bootstrap draws, and
prototype predictions are immutable. Only a `Linear(256, 7)` head is fitted. Zero
labels remains the frozen source classifier; positive budgets are target-assisted.

## Primary readout

The head starts from the corresponding source-trained `network.13` parameters and is
optimized in float64 by deterministic full-batch LBFGS. The loss is balanced mean
support cross-entropy plus `100 / 1799` times squared displacement from the frozen
source head. The value 100 was selected once from the frozen P1--P3 FS4 source-only
candidate summary using the preregistered global rule and is fixed for every budget.
No P4 query data participate in fitting, stopping, selection, or regularisation.

## Endpoints and uncertainty

Primary endpoint: Macro-F1 difference between linear and cosine-prototype predictions
at every identical positive budget. Secondary endpoints include Accuracy, balanced
accuracy, per-class/worst-class F1, support fit, held-condition query performance,
condition performance, confusion matrices, and convergence diagnostics.

Uncertainty uses exactly 10,000 TagID-stratified physical-block bootstrap draws crossed
with source and support seed resampling. Linear-minus-prototype and linear-minus-zero
intervals are formed by subtracting metrics within each matched bootstrap replicate.

The practical threshold is +0.05 Macro-F1. Reliability requires the paired 95% CI
lower bound to be greater than zero.

## Classification precedence

1. `LINEAR_READOUT_UNSTABLE_OR_NON_IDENTIFIABLE` if a critical validity, freeze,
   determinism, prediction-completeness, or finite-optimization gate fails.
2. `LINEAR_READOUT_RESCUES_TARGET_CALIBRATION` if a budget of at least 35 labels has
   both a reliable/practical linear-minus-prototype gain and a reliable/practical
   linear-minus-zero gain.
3. `PARTIAL_LINEAR_READOUT_BENEFIT` if rule 2 is not met but any positive budget has
   a paired linear-minus-prototype CI above zero or a reliable/practical gain over zero.
4. `REPRESENTATION_BOTTLENECK_SUPPORTED` otherwise.

These are diagnostic, not causal, statements. Encoder fine-tuning is not part of this
experiment and will not be executed.
