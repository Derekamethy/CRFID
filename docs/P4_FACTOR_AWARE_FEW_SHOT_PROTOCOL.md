# P4 Factor-Aware Few-Shot Protocol

The study tests one isolated explanation for the historical P4 Few-Shot result:
whether labelled supports failed to cover ER and surface factors present in
unseen query conditions. It compares support strategies at exactly matched
1/3/5-shot budgets while preserving the five frozen source-only C1 encoders,
fixed adaptation methods, query folds, and support seeds.

Three shared Latin-square folds query `E0xS0/E1xS1/E2xS2`,
`E0xS1/E1xS2/E2xS0`, and `E0xS2/E1xS0/E2xS1`. Every fold queries 21 complete
TagID-by-factor blocks, including all 50 repetitions, and leaves six candidate
cells per TagID. Across folds, all 63 physical blocks are queried once.

One block-shot is one labelled row from one unique support block. Random support
samples distinct cells uniformly. At 3-shot, named balanced strategies cover all
levels of ER, surface, or both; joint coverage is a perfect three-cell matching.
At 5-shot, joint coverage minimises ER imbalance, then surface imbalance, then
maximises pairwise Manhattan dispersion. Fixed seeds break remaining ties. Every
class uses the same selected factor-cell set, and a SHA-256-derived fixed offset
selects one of the 50 rows in each chosen support block.

The descriptive coverage score is
`0.25*(unique_ER/3 + unique_surface/3) + 0.5*(mean pairwise Manhattan distance/4)`.
It is recorded for exploratory coverage-response analysis and is not an
optimisation endpoint.

The primary adapter L2-normalises support embeddings, averages them within
class, re-normalises each prototype, and predicts by cosine similarity. The
secondary adapter reuses the released float64 full-batch LBFGS proximal linear
head with its fixed 1/3/5-shot lambdas. The source encoder never changes. FS0
uses the frozen source head on the identical query folds.

Query labels are held by a fail-closed seal. Each episode selects support,
constructs the adapter, freezes and hashes query predictions, opens labels, and
computes metrics once. The primary endpoint is out-of-fold block Macro-F1 after
deterministic majority vote. A 10,000-replicate paired TagID-stratified block
bootstrap supplies the primary interval; repeated rows are not inferential
units. Full settings are frozen in
`configs/p4_factor_aware_few_shot/canonical.json`.
