# P4 Factor-Aware Few-Shot Preregistered Protocol

This study tests whether matched-budget factor-aware support coverage improves P4
Few-Shot adaptation. It uses only P4, the five frozen source-only
`C1_FIRST_DIFFERENCE_ERM_1DCNN` checkpoints (seeds 42-46), and five fixed
support-selection/support-row seeds.

The indivisible unit is `TagID x ER x surface x P4`. Three shared Latin-square
outer folds hold out three factor cells per TagID; each held fold contains every
ER and every surface, all 50 repetitions remain together, and all 63 blocks are
queried exactly once. A block-shot is one labelled row from one distinct support
block. Budgets are therefore 7, 21, and 35 total labels at 1-, 3-, and 5-shot.

The primary endpoint is 63-block out-of-fold Macro-F1. The primary contrast is
3-shot `TARGET_ONLY_COSINE_PROTOTYPE` `JOINT_FACTOR_COVERAGE - RANDOM_BLOCK`,
paired by outer fold, source seed, support seed, query blocks, and label budget.
The prototype normalises each support embedding, averages within class,
normalises the class prototype, and predicts by cosine similarity. The secondary
head reuses the released float64 full-batch LBFGS proximal recipe and fixed
lambdas (100, 100, 0.1); only a copy of `network.13` is adapted.

`RANDOM_BLOCK` samples candidate cells uniformly without replacement.
At 3-shot, ER- and surface-balanced strategies cover their named factor, while
joint coverage uses a three-cell perfect matching covering all ERs and surfaces.
At 5-shot, joint coverage lexicographically minimises ER imbalance, minimises
surface imbalance, maximises summed pairwise Manhattan dispersion on the 3x3
factor grid, then uses the fixed seed to break ties. The same cell set is used
for all seven classes. Support-row selection is a fixed SHA-256-derived offset
within each 50-row block.

The descriptive support-coverage score is fixed as
`0.25*(unique_ER/3 + unique_surface/3) + 0.5*(mean_pairwise_Manhattan/4)`.
It is not an optimisation endpoint and cannot alter the primary comparison.

Query labels remain in a fail-closed seal until predictions are frozen and
SHA-256 hashed. Metrics are computed exactly once after the seal opens. Physical
blocks are reduced by majority vote with the lowest frozen class index breaking
an exact tie. Uncertainty uses 10,000 deterministic TagID-stratified paired block
bootstrap replicates; row repetitions are never inferential units. The head
support-fit validity threshold, recorded without retuning, is Accuracy >= 0.95
and Macro-F1 >= 0.95.
