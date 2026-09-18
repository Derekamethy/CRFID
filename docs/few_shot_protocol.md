# Few-Shot P4 Protocol

## Source dependency

The representation is the penultimate 256-dimensional output (`network.12`) of
the frozen source-only C1 first-difference 1D-CNN trained on P1-P3 with seeds
42-46. Encoder weights, the source head (`network.13`) and the first-difference
preprocessing state are frozen. Inputs are 280-point first differences of the
281-point raw signals, standardized with the source-only state only.

## Episodes, support and query

Nine `(surface, ER)` combinations are ordered lexicographically and arranged into
18 preregistered episodes from two cyclic families. Five combinations form the
support pool; the remaining four form a 1,400-sample query set. The P4 structure
is 3 files, 3 surfaces, 3 ER values, 7 classes, 63 complete condition blocks and
3,150 samples, with 50 readings per block.

One shot is one labelled P4 measurement per class, drawn from distinct complete
condition blocks. Frozen support sizes are 0, 7, 21 and 35 for 0, 1, 3 and
5-shot. Support sets are class-balanced and nested. The query set is identical
across shot conditions inside an episode.

Every support-pool block is excluded from query in full, including unused pool
blocks. Isolation is enforced on sample identity, condition block, repeat group,
exact signal digest and source data row; 54 formal episode-by-shot isolation
checks were recorded and all passed.

Support ranking is a SHA-256 over a frozen salt, class, surface, ER and canonical
sample identity. Signal magnitude, model output and query outcomes are excluded
from the ranking by construction.

## Adaptation methods

**Target-only prototypes.** For each seed, episode, shot and class, the support
embeddings are averaged in float64 after ordering supports by ascending canonical
sample identity. Prediction uses float64 squared Euclidean distance with no
normalization, no square root and no temperature. `argmin` runs in class-index
order, so an exact tie selects the lowest class index. Source prototypes,
source-target mixing and fine-tuning are prohibited.

**Source-anchored linear head.** The frozen source head is re-fitted on the
support set in float64 with L-BFGS (`max_iter` 100, history 20, strong-Wolfe line
search) minimising support cross-entropy plus a proximal penalty
`lambda * ||theta - theta_source||^2 / 1799` that anchors the solution to the
source head. Only `network.13` changes; the encoder is untouched.

## Selection and freeze rules

The regularization strength was selected from `{0.01, 0.1, 1.0, 10.0, 100.0}` on
P1-P3 pseudo-target evidence only (stage FS4), before any P4 head execution. No
P4 signal, P4 support content, P4 metric, sealed label or prototype-branch metric
entered that selection. Selected values are 100 at 1-shot, 100 at 3-shot and 0.1
at 5-shot.

## Label boundaries

Support labels are available to adaptation. Query labels were sealed before any
numerical P4 access, are absent from every adaptation-visible manifest, and were
opened only after all 360 Stage-A prediction units for a stage had been written
and hashed. Prediction regeneration after label access is prohibited. No
transductive use of query samples is declared or performed.

## Metrics and aggregation

Per unit: sample accuracy, sample macro-F1, condition-level and
unique-signal-weighted variants, worst-class recall, zero-recall class count,
dominant predicted-class fraction and mean prediction entropy. Macro-F1 is the
unweighted mean of the seven per-class F1 values; classes with no support or no
prediction contribute zero rather than an undefined value.

Aggregation is a mean over the five seeds within each episode, then an
equal-weight mean over the 18 episodes. Reported dispersion is the population
standard deviation over the 18 episode means. Uncertainty intervals use a
deterministic paired-episode percentile bootstrap with 10,000 resamples and seed
20260720.

Class order is fixed as indices 0-6 mapped to Tag-1 through Tag-7 in every table.
