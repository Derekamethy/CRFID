# Fixed-position condition-grouped learning-validity protocol

Status: **FROZEN BEFORE TRAINING** on 2026-08-03. Protocol identifier:
`FIXED_POSITION_CONDITION_GROUPED_VALIDITY_V1`. The machine-readable declaration
is `configs/fixed_position_grouped_validity/canonical.json`. No optimizer step is
permitted until this file, the data audit, the exact split manifest, and all ten
pre-training validation gates exist.

## Question and scope

The benchmark asks whether the existing end-to-end TagID classifier learns on
previously unseen physical condition blocks when measurement position is fixed.
P1, P2, P3, and P4 are separate benchmarks. A benchmark run receives rows from
exactly one position. No target-assisted method, cross-position training,
adaptation, CORAL, SSL, NCM, prototype method, model search, or alternative
architecture is permitted.

## Governed inputs and group definition

The raw data root is selected only by `CRFID_FIXED_POSITION_DATA_ROOT`; the
canonical Strict-DG artifact root is selected only by
`CRFID_STRICT_DG_ARTIFACT_ROOT`; large runtime state is selected only by
`CRFID_FIXED_POSITION_RUN_ROOT`. Raw measurements, full prepared arrays,
checkpoints, logits, predictions, embeddings, and caches remain outside Git.

One indivisible group is `(TagID, ER, surface, fixed_position)`. Since position
is fixed within a run, its effective key is `(TagID, ER, surface)`. Every group
must contain exactly 50 rows. Each position must contain seven TagIDs, three ER
levels, three surfaces, 63 groups, and 3,150 rows. Any mismatch stops scientific
execution with `BLOCKED_GOVERNED_INPUTS` or `DATA_STRUCTURE_FAILURE`.

The canonical source loader, registry, labels, and raw P4 loader are reused.
Raw TagID 1--7 must map through `crfid.data.tyndall.map_raw_tag_label` to class
indices 0--6. The measurement carrier is the ordered 281-position vector whose
columns are `0` through `280`; the canonical release explicitly does not claim
a resolved physical frequency axis.

## Frozen split method

There are three outer folds per position. Give surfaces A1, A2, and A3 indices
0, 1, and 2. A block is assigned to outer fold
`(ER + surface_index) modulo 3`. Thus every test fold has exactly one block from
each ER and each surface for each TagID: three blocks per TagID, 21 blocks and
1,050 rows total. The three test folds partition all nine blocks of every TagID.
Outer test membership is independent of seed.

For each position, fold, seed, and TagID, the six non-test blocks are sorted and
permuted by a SHA-256-derived deterministic seed scoped by protocol identifier,
position, fold, seed, and TagID. The first two blocks are validation and the
remaining four are training. Every run therefore has 28 training blocks (1,400
rows), 14 validation blocks (700 rows), and 21 held-condition blocks (1,050
rows), with all seven classes present in every partition. Seeds are 42, 43, 44,
45, and 46. Total primary runs are 4 x 3 x 5 = 60.

The sanitized exact manifest is written before training. Each row records
position, fold, seed, TagID, ER, surface, hashed condition-block identifier,
row count, split assignment, and source-data fingerprint. It contains no sample
identifier or signal value.

## Frozen model, preprocessing, and training

The only primary model is `C1_FIRST_DIFFERENCE_ERM_1DCNN`, implemented by the
canonical `NeutralSourceOnlyCNN1D` (142,855 parameters). Initialization,
deterministic permutation, loss, optimizer, inference, metrics, and checkpoint
loading reuse the canonical Strict-DG implementation.

Preprocessing is fitted only on the permitted training partition: first
difference `x[i+1]-x[i]` without padding (281 to 280), featurewise population
mean and standard deviation in float64, the canonical future-execution rule
`std < 1e-12 -> 1.0`, then float32 conversion. For outer refit, preprocessing is
refitted on train plus validation only. Held blocks never affect preprocessing.

Training uses unweighted, unsmoothed cross entropy; AdamW with learning rate
0.001, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8, AMSGrad false,
foreach false, and fused false; batch size 256; no scheduler. Inner training is
at most 50 epochs with patience 8. A checkpoint improves only when validation
Macro-F1 exceeds the previous best by more than 1e-12, so the earliest tied
epoch remains selected. The model is then reinitialized with the same seed and
refitted on train plus validation for exactly the selected epoch count. Held
conditions are evaluated once after checkpoint reload. No hyperparameter varies
by position, and P4 has no selection role beyond its own training-only and
validation-only partitions.

## Pre-training validation gates

Before training, execution must prove: (1) no block occurs in more than one
split; (2) all 50 repeats remain together; (3) every test fold contains all
seven TagIDs; (4) per-split class and row counts are exact; (5) label mappings
match Strict-DG; (6) ordered carrier and first-difference dimensions match;
(7) run partitions contain only their named position; (8) no target-assisted
or cross-position row is supplied; (9) manifests reproduce byte-for-byte for a
fixed seed; and (10) changing a seed leaves test membership fixed, changes only
validation/training membership and training randomness, and changes at least
one validation assignment across the benchmark.

## Sanity checks and failure criteria

Every primary run records initial held performance, final train-plus-validation
Accuracy and Macro-F1, selected-checkpoint validation performance, held
performance, learning curves, prediction histogram, and train-to-held gap.

One diagnostic per position uses only the fold-0, seed-42 training split. It
selects the first eight rows per class after stable row ordering (56 rows, one
batch), initializes the canonical model at seed 42, and performs at most 250
full-batch AdamW steps with the frozen primary optimizer and loss. It passes
only if true-label subset Accuracy and Macro-F1 both reach at least 0.95. The
diagnostic does not alter a split, primary hyperparameter, epoch selection, or
checkpoint. A failing position is classified `LEARNING_VALIDITY_FAILURE`; its
held-condition metrics remain reported but are not interpreted as domain-
generalisation evidence.

Any structure, leakage, mapping, dimension, isolation, determinism, or runtime
failure invalidates the affected execution. Missing governed inputs yield
`BLOCKED_GOVERNED_INPUTS`; values are never simulated or fabricated.

## Metrics, baselines, and confidence intervals

For each run report row Accuracy and Macro-F1; mean-logit condition-block
Accuracy and Macro-F1; per-class precision, recall, and F1 at row and block
levels; confusion matrices; predicted-class histograms; final training,
validation, and held Accuracy; and the train-to-held Accuracy gap. Baselines are
uniform seven-class chance (1/7), empirical row majority, and condition-block
majority, all computed rather than assumed.

For each position and metric report the mean, population standard deviation,
median, minimum, and maximum across its 15 runs. Held-metric 95% intervals use
10,000 percentile bootstrap resamples with seed 20260803. The sampling unit is
the condition block, stratified by TagID; a selected block retains all 50 rows
and its paired predictions from all five seeds. No interval treats repeated
rows as independent.

## Interpretation rule

No numerical definition of “strong” will be introduced after results are seen.
The report will present effect sizes, block-based intervals, baselines, and
prediction distributions, then use pattern A, B, C, or D from the task
specification only as evidence-supported descriptive language. A failed
learning-validity position cannot support a generalisation claim.

## Expected committed outputs

The compact result directory must contain `00_EXECUTIVE_SUMMARY.md`,
`01_SOURCE_REPOSITORY_PRESERVATION.json`, this protocol,
`03_DATA_AND_GROUP_STRUCTURE_AUDIT.csv`, `04_SPLIT_VALIDATION_RESULTS.csv`,
`05_RUN_REGISTER.csv`, `06_PER_RUN_METRICS.csv`,
`07_POSITION_AGGREGATES.csv`, `08_PER_CLASS_RESULTS.csv`,
`09_PREDICTED_CLASS_HISTOGRAMS.csv`,
`10_CONFUSION_CONSISTENCY_REPORT.md`, `11_LEARNING_VALIDITY_RESULTS.csv`,
`12_INTERPRETATION.md` and `STATUS.md`.
`learning_curves.csv` and compact confusion matrices are also expected.
The exact split manifest and its SHA-256 sidecar are committed under
`manifests/fixed_position_grouped_validity/`; large runtime artifacts are not.
