# Paired 2x2 angle-distance factorial contrast protocol

Status: preregistered before reconstruction of checkpoint predictions or calculation of final factorial estimates.

## Scope and frozen inputs

This diagnostic uses retained fixed-position evidence and the frozen synchronized
Case-B design. The four measurement cells are P1 = 50 mm/0 degrees,
P2 = 50 mm/45 degrees, P3 = 150 mm/0 degrees, and P4 = 150 mm/45 degrees.

Public preflight validates governed measurement fingerprints and the fixed
configuration without training. Execution reconstructs compact reference
aggregates and runs the 60 synchronized C1 units, verifying inference against
their checkpoint-bound metrics. Historical Git commits, tags and repository
snapshots are not scientific execution inputs. See [reproducibility](../REPRODUCIBILITY.md).

## Case B paired design

The pairing analysis found that the inherited outer test assignments are
position-synchronized, but 492 `physical cell x fold x seed` groups have
different train/validation assignments across positions. The inherited
validation RNG included the position name. The existing 60 runs therefore fail
the full pairing requirement and will not be used for factorial inference.

Governed raw inputs and the exact frozen C1 training configuration are
available, so Case B is preregistered before any synchronized rerun. Exactly 60
new primary runs will use the existing `C1_FIRST_DIFFERENCE_ERM_1DCNN`, the same
three outer folds, the same seeds 42-46, and no hyperparameter tuning or model
comparison. Test membership remains the original balanced rule. Within each
TagID, two validation blocks are selected from the six non-test blocks using
deterministic RNG material
`PAIRED_ANGLE_DISTANCE_FACTORIAL_CONTRAST_V1|fold={zero_based_fold}|seed={seed}|tag={TagID}`.
Position is deliberately excluded, so train, validation, and test membership is
identical at P1-P4. The completed four-position learning-validity evidence is
reused without rerunning or tuning that diagnostic.

## Estimands and contrasts

The primary inferential unit is one of 63 physical condition blocks. The 50 rows
inside a block are correlated repetitions and are never independent inferential
units. For each physical block, seed, and position, the block prediction is the
unchanged grouped-validity rule: the canonical condition-level prediction
returned by the frozen evaluation implementation. Each cell appears in the held
set of exactly one outer fold for every seed.

The primary endpoint is paired condition-block accuracy from the synchronized
Case B rerun. Its position estimand
is the mean of the five seed-specific accuracies pooled over all 63 held blocks.
For any position metric `M`:

- angle effect = `0.5 * ((M_P2 - M_P1) + (M_P4 - M_P3))`;
- distance effect = `0.5 * ((M_P3 - M_P1) + (M_P4 - M_P2))`;
- interaction = `(M_P4 - M_P3) - (M_P2 - M_P1)`.

Negative angle and distance effects mean lower performance at 45 degrees and
150 mm, respectively. The interaction sign is the 150-mm angle effect minus the
50-mm angle effect.

Macro-F1 is nonlinear and no per-block Macro-F1 will be invented. For each
position and seed, the 63 aligned held-block predictions are pooled, Macro-F1 is
computed, and the five seed-specific values are averaged before a factorial
contrast is formed. This pooled-block estimand remains distinct from the
descriptive mean of 15 fold-level run Macro-F1 values.

## Uncertainty

The primary interval is a two-sided 95% percentile interval from 10,000 paired,
TagID-stratified physical-block bootstrap replicates using random seed 20260804.
Within every TagID, its nine `ER x surface` blocks are sampled with replacement;
the selected block retains all four positions and all five seeds. The sensitivity
analysis additionally samples the five seeds with replacement once per replicate
and applies the same sampled seed indices to every block and position. Rows are
never resampled as independent units. P-values are not reported.

## Secondary and exploratory analyses

Secondary endpoints are pooled condition-block Macro-F1, descriptive row
accuracy, pooled per-class recall and F1, predicted-class distribution,
train-to-held and validation-to-held accuracy gaps, early-stopping epoch, and
run-level variance. Prespecified heterogeneity summaries cover TagID, ER,
surface, angle separately at each distance, and distance separately at each
angle. These summaries are not searched for a preferred narrative.

## Interpretation boundary

The final classification uses only the five categories requested in the task.
Angle- or distance-associated degradation requires a negative primary contrast
whose 95% interval excludes zero and directionally consistent simple effects.
An interaction requires an interaction interval excluding zero with compatible
simple effects. Otherwise the result is mixed or not confirmed. Language is
associational, not causal. Passed tiny-set learning validity is reported
separately from held-condition generalisation and factor-associated differences.
