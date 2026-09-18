# Few-Shot P4 Adaptation

Status: `MIGRATED_FROZEN_BRANCH_COMPLETE_VERIFIABLE_EXECUTION_PERMANENTLY_CLOSED`

Scientific label: `CONDITION_DISJOINT_FEW_SHOT_TARGET_DOMAIN_ADAPTATION_FROM_FROZEN_SOURCE_MODELS`

## Purpose

This branch measures how much a small amount of **labelled target-domain (P4)
calibration data** recovers the accuracy that the strict source-only model loses
when it is transferred to P4. It answers a different question from the strict
domain-generalization branch: strict DG asks what happens with *no* target
labels, this branch asks what happens with 1, 3 or 5 labelled measurements per
class.

Because P4 labels are used, this branch is **domain adaptation, not domain
generalization**. Neither "strict untouched target" nor "prospective unseen
target" applies to any number reported here.

## Dependency on the frozen source model

Adaptation starts from the frozen source-only C1 first-difference 1D-CNN
(`C1_FIRST_DIFFERENCE_ERM_1DCNN`), trained on P1-P3 only, with the five frozen
seeds 42-46. Encoder weights, the first-difference preprocessing state and the
source classifier head are all frozen; nothing in this branch retrains the
encoder.

The exact source artifacts consumed by the original execution were copied
byte-for-byte into the branch itself, under
`frozen_branch/01_frozen_source_import/frozen_last_code_snapshot/`. Verification
of this branch therefore reads **only** files inside the branch and never
depends on the strict DG output tree, on its finalisation package, or on any
path outside this repository.

## Zero-shot versus adaptation

| Curve | Meaning | Target labels used |
|---|---|---|
| `FROZEN_SOURCE_HEAD`, shot 0 | The frozen source classifier applied directly to the P4 query set | none |
| `TARGET_ONLY_SQUARED_EUCLIDEAN_PROTOTYPES`, shots 1/3/5 | Class prototypes built **only** from the labelled P4 support set; prediction by squared Euclidean distance | support only |
| `FROZEN_ENCODER_SOURCE_ANCHORED_LINEAR_HEAD_ADAPTATION`, shots 1/3/5 | The frozen source head re-fitted on the support set with a proximal penalty anchoring it to the source solution | support only |

The zero-shot row is a **shared reference**, evaluated once on the same query
population and reused by both adaptation curves. It is not two separate results.

## Shot definitions

One shot is **one labelled P4 measurement per class**. Support measurements come
from distinct complete condition blocks; a block is identified by tag, position,
surface, ER and canonical source-condition identity.

| Shot | Labelled support samples per episode | Query samples per episode | Support as a fraction of query |
|---:|---:|---:|---:|
| 0 | 0 | 1400 | 0.0% |
| 1 | 7 | 1400 | 0.5% |
| 3 | 21 | 1400 | 1.5% |
| 5 | 35 | 1400 | 2.5% |

Support sets are **class-balanced** (exactly `shot` samples per class, 7 classes)
and **nested**: the 1-shot support set is a subset of the 3-shot set, which is a
subset of the 5-shot set. The query population is identical across all four shot
conditions within an episode, so shot-to-shot deltas are paired comparisons on
the same samples.

## Support and query separation

Nine `(surface, ER)` combinations are sorted lexicographically and arranged into
18 preregistered episodes using two cyclic families. Five ordered combinations
form the support pool and the remaining four form the 1,400-sample query.

Every support-pool condition block is excluded from query **in full**, including
pool blocks that no shot condition actually uses. Isolation was frozen and
verified on five independent identities:

- sample identity;
- condition block;
- repeat group;
- exact signal digest;
- source data row.

Support ranking uses a SHA-256 over a frozen salt, the class, the surface, the
ER value and the canonical sample identity. No signal magnitude, model output or
query outcome enters the ranking.

## Selection and evaluation boundaries

**Allowed target information.** Support samples and their labels, and the raw P4
signal columns needed to embed support and query samples.

**Prohibited query-label use.** Query labels were sealed before any P4 access and
were not available to adaptation, hyperparameter selection, early stopping,
checkpoint selection, seed selection, episode design or preprocessing. The
adaptation-visible query manifest carries **no label column**.

**Model and hyperparameter selection.** The proximal-head regularization strength
was chosen from the grid `{0.01, 0.1, 1.0, 10.0, 100.0}` on **source-side
pseudo-target evidence from P1-P3 only** (stage FS4). No P4 signal, no P4
support content, no P4 metric and no prototype-branch metric entered that
choice.

**Evaluation timing.** Each execution stage ran in two parts. Stage A generated
and hash-sealed every prediction, prototype, adapted head and class matrix while
the query labels were still sealed. Only after that closure did Stage B open the
sealed labels exactly once and compute metrics. Prediction regeneration after
label access is prohibited and did not occur.

## Formal evaluation units

A formal unit is one `(checkpoint seed, episode, shot)` evaluation on the full
1,400-sample query set.

```
18 episodes x 5 seeds x 4 shot conditions = 360 units per method line
  of which  18 x 5 x 1 =  90 zero-shot reference units
  and       18 x 5 x 3 = 270 adaptation units
```

Both method lines (prototype and source-anchored head) contain 360 units, and
their 90 zero-shot units are byte-identical to each other. The 360 units of each
line match one-to-one on query identity. Counting distinct evaluations rather
than per-line rows gives 630 = 90 shared zero-shot + 270 prototype + 270 head.

## Aggregation

Metrics are computed per unit, averaged over the five seeds **within** an
episode, then averaged with equal weight over the 18 episodes. The reported
dispersion is the **population standard deviation over the 18 episode means**,
not a pooled standard deviation over the 90 units.

## Primary results

Sample accuracy and macro-F1, mean over 18 episodes with population SD:

| Method | Shot | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Frozen source head (zero-shot) | 0 | 0.156635 (0.023472) | 0.107257 (0.025602) |
| Target-only prototypes | 1 | 0.142698 (0.037640) | 0.112159 (0.033743) |
| Target-only prototypes | 3 | 0.165429 (0.021727) | 0.145141 (0.023730) |
| Target-only prototypes | 5 | 0.162635 (0.029901) | 0.144360 (0.029465) |
| Source-anchored linear head | 1 | 0.147476 (0.031605) | 0.120852 (0.027799) |
| Source-anchored linear head | 3 | 0.149754 (0.024019) | 0.129295 (0.021783) |
| Source-anchored linear head | 5 | 0.157651 (0.040010) | 0.138984 (0.036008) |

The prototype 3-shot row is the best observed value among the separately
evaluated methods. It is a descriptive maximum, **not** a deployable combined
adapter; no per-shot or per-class hybrid is claimed anywhere in this branch.

Paired episode deltas over macro-F1, with deterministic percentile bootstrap
intervals (10,000 resamples, seed 20260720):

| Comparison | Mean delta | 95% interval | Positive / tie / negative episodes |
|---|---:|---:|---:|
| Prototype 0 to 3 | 0.037884 | [0.022224, 0.056097] | 17 / 0 / 1 |
| Prototype 0 to 5 | 0.037103 | [0.019806, 0.054516] | 16 / 0 / 2 |
| Head 0 to 3 | 0.022038 | [0.011519, 0.032641] | 16 / 0 / 2 |
| Head 0 to 5 | 0.031727 | [0.012092, 0.050825] | 14 / 0 / 4 |
| Head minus prototype at 3 | -0.015846 | [-0.031515, -0.000718] | 6 / 0 / 12 |

Both adapters improve macro-F1 over zero-shot at 3 and 5 shots. Accuracy stays
close to the zero-shot level throughout: the gain is a redistribution across
classes, not a large overall accuracy jump.

## Limitations

- Absolute performance remains low. Adapting with 0.5-2.5% labelled target data
  does not restore the accuracy the source model reaches inside its own domains.
- The two adapters were evaluated **separately**. The per-shot winner changes,
  and selecting the winner per shot after seeing P4 results would be a
  post-hoc choice, which is why no combined adapter is claimed.
- Source-side selection did not predict the target ranking. Stage FS4 preferred
  the head at every shot on P1-P3 evidence; on P4 the head won only at 1-shot.
  This is evidence that P4 is a materially different shift, not a tuning error.
- The bootstrap intervals are supporting descriptive uncertainty over 18 paired
  episodes; they do not establish universal superiority of either adapter.
- Head displacement and support cross-entropy evidence indicates a support-set
  overfitting risk, most clearly at 5-shot with the low-regularization recipe.
  The association is descriptive and does not establish causality.
- The historical target-informed development figure of roughly 0.577 accuracy is
  a `RETROSPECTIVE_TARGET_INFORMED_DEVELOPMENT_REFERENCE`. It is not comparable
  to any number in this branch and is not a few-shot result.

## Reproducibility scope

Compact results are retained under [few_shot](../../results/canonical_metrics/few_shot/). Full historical verification requires an external frozen few-shot archive containing the original per-episode records, manifests, checkpoint imports and seals. That archive is not included in the public repository; measurements alone are insufficient. The retained verification scripts describe that archive contract, and the shared `run.py` entry point is a planner.

The scientific result remains a within-corpus labelled-target calibration study with condition-block-disjoint support/query episodes. It is not an independent deployment campaign or a source-only result. See [reproducibility](../../REPRODUCIBILITY.md) and the [protocol](../../docs/few_shot_protocol.md).
