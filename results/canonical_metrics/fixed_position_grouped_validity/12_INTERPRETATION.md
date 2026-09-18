# Interpretation

## Direct answers

**Did the benchmark run fully?** Yes. All 60 preregistered C1 runs completed,
with exactly 15 runs for each fixed position. All structure and leakage gates,
checkpoint reloads, one-time held evaluations, confusion checks, and compact
metric recomputations passed.

**Did each position pass basic learning validity?** Yes. Each canonical model
fit its position's true-label 56-row training-only diagnostic subset above the
preregistered 0.95 Accuracy and Macro-F1 thresholds.

**Does within-position unseen-condition TagID generalisation exist?** The
answer is position-specific. P1 and P3 provide above-baseline evidence. P2 and
P4 do not provide clear evidence beyond the balanced baseline. It would be
incorrect to collapse these results into either “all positions learn” or “all
positions fail.”

**Is P4 uniquely difficult?** No. P4 held row Accuracy is 0.1711, but P2 is
lower at 0.1499. Their held row Macro-F1 means are 0.0861 and 0.0885,
respectively. Both positions' block-bootstrap intervals overlap the 1/7
baseline.

## Evidence-supported pattern

This is interpretation **D: mixed positions**.

P1 has the highest held row Accuracy mean (0.2688) and a block-cluster interval
of [0.2147, 0.3257]. P3 is next at 0.2389 [0.1902, 0.2927]. Their pooled-block
Macro-F1 intervals, [0.1788, 0.2846] and [0.1631, 0.2649], also lie above 1/7.
That is evidence that some TagID structure survives held ER/surface conditions
within those fixed positions.

P2 held row Accuracy is 0.1499 [0.1081, 0.1948] and P4 is 0.1711 [0.1300,
0.2154]. Their pooled-block Macro-F1 intervals are [0.0771, 0.1581] and
[0.0929, 0.1699]. These intervals overlap chance, and both positions have low
Macro-F1, so the evidence does not support a reliable unseen-condition claim
for either.

The prediction distributions are not complete single-class collapse, but they
are uneven. The largest aggregated predicted-class share is 31.8% for TagID 1
at P3, 23.6% for TagID 6 at P4, 22.2% for TagID 5 at P2, and 20.1% for TagID 4
at P1. Per-class and per-run detail is retained in the compact result tables.

## Learning versus generalisation

The sanity checks distinguish basic learnability from condition transfer. P1,
P2, P3, and P4 all fit their tiny true-label subsets, but primary mean
train-to-held Accuracy gaps are 0.4247, 0.4737, 0.5344, and 0.3037. Thus the
weak P2/P4 held results cannot be attributed simply to a model that never
learned its training labels. They instead indicate that the fitted decision
rules are poorly stable across unseen ER/surface blocks under this protocol.

## Limitations and non-claims

- The condition-block population is 63 blocks per position: nine physical
  conditions per TagID. Intervals quantify uncertainty over these observed
  blocks, not arbitrary future experimental regimes.
- The 50 repeated rows within a block are correlated. They remain grouped in
  splitting and resampling, and row counts are not used to claim significance.
- Macro-F1 is nonlinear. The reported descriptive mean averages 15 fold/seed
  values; its block-bootstrap interval targets the mean of five seed metrics
  after pooling all 63 held blocks per seed. The table labels both estimands.
- Early-stopped epoch counts vary substantially (1 to 27), and run-level
  performance is variable. Conclusions should use the full distributions, not
  a single seed or fold.
- No physical frequency vector is asserted. The benchmark uses the canonical
  ordered position indices 0--280 and first differences of length 280.
- The result does not test new TagIDs, a new dataset, cross-position transfer,
  adaptation, or causal physical mechanisms.
- No additional model family or P4-informed hyperparameter selection was run.

Overall validation assessment: **share with caveats**. The executed methods and
compact calculations are internally consistent and answer the preregistered
question, but the limited condition population, estimator distinction for
Macro-F1, and substantial run variability must remain visible.
