
# Limitations

- The same underlying 3,150 P4 samples are reused across split seeds; the 25
  paired units are not fully independent observations.
- P4-Adapt and P4-Val contain repeated measurements from already-known P4
  conditions.
- This experiment measures within-P4 target assistance, not generalisation to a
  newly collected P4 acquisition campaign.
- The encoder remains frozen in A2; any effect is a readout/adaptation effect,
  not evidence that target labels improved the encoder.
- Results do not establish performance on a new laboratory session, device, tag
  manufacturing batch, or environment.
- The historical retrospective P4 result remains methodologically separate and
  non-comparable.
- No post-result method, candidate, or hyperparameter expansion was permitted.
- Candidate-set breadth limits the conclusions.
- Aggregate confusion counts repeat samples across source and split seeds.
- A negative or mixed outcome remains a scientifically valid execution.

## Data dependence and near-duplicate structure in P4 (required disclosure)

Historical dependence diagnostics report the following results. Their full replay material is not distributed in the public repository.

- The P4 dataset has **3,150 rows** but only **1,076 exact-unique bitwise
  spectra** — a **2,074-row duplicate excess**. 942 of 997 multi-row duplicate
  groups are exact triplicates; only 79 of 3,150 rows are singleton spectra.
- Exact-signal SHA-256 groups did **not** cross Adapt/Val/Test roles in any of
  the five split seeds (verified 0 crossings), but exact grouping alone did
  **not** remove near-duplicate dependence between roles.
- Approximately **93–95% of P4-Test rows** had cosine similarity above 0.999
  to some P4-Adapt row (historical dependence diagnostics).
- Approximately **98.4% of P4-Test rows'** nearest P4-Adapt neighbour lay in
  their own TagID × ER × surface condition cell.
- A **random, untrained encoder** control and a **raw-signal, no-encoder**
  control matched or exceeded the trained Strict-DG encoder's A2 linear-probe
  and 1-NN accuracy under the identical within-condition split protocol (probe
  accuracy: trained 0.9835/0.9875, random-untrained 0.9782, raw-signal
  0.9946; 1-NN accuracy: random-untrained 0.9909, raw-signal 0.9990).
- **Leave-one-ER-out** accuracy was approximately chance (0.1432 vs a chance
  rate of 0.1429).
- **Leave-one-condition-cell-out** accuracy was below chance (0.0865).
- The random-encoder/raw-signal controls and the leave-ER-out /
  leave-condition-cell-out diagnostics were run on source seed 42 only (5
  split seeds each); the effect size is large enough that seed variation is
  considered very unlikely to change the conclusion, but the other four source
  seeds were not run through this specific ablation.

Threshold-dependent diagnostic clustering suggested that the effective number
of strongly separated local measurement groups may be far smaller than the
nominal 3,150 rows (in the historical diagnostics: agglomerating at 1% of the median pairwise
distance collapsed all 3,150 rows into 133 clusters; at 5%, into 122 clusters;
every cluster was label-pure and condition-pure). This is a
threshold-dependent diagnostic estimate, not an exact physical sample count.

### Required interpretation of the A2 results (supersedes any bare citation of 0.9835 elsewhere)

The frozen-encoder linear probe achieved 0.9835 Accuracy and 0.9835 Macro-F1
under the preregistered within-condition P4 split. The result is numerically
valid and free from implementation-level label leakage, but it is materially
inflated by strong repeated-measurement and near-duplicate dependence.
Random-encoder and raw-signal controls achieved comparable or higher
performance, while leave-ER-out and leave-condition-cell-out diagnostics fell
to chance or below. Therefore the result should be interpreted as within-P4,
condition-covered calibration rather than evidence of transferable
representation learning or generalisation.

The within-condition A2 score does not establish that the trained Strict-DG
encoder contains uniquely useful or transferable P4 TagID information.
Comparable results from random-encoder and raw-signal controls show that the
score is primarily supported by the local structure and dependence of the P4
measurements under the condition-covered split.

Target-fitted readouts (A2) strongly outperform source-fitted readouts (A0)
within condition-covered P4 splits, but the present data cannot determine
whether this reflects a transferable decision-boundary shift or interpolation
among condition-specific, highly dependent measurements. This branch does not
establish decision-boundary misalignment as the dominant failure mechanism.

## Historical retrospective P4 — explicit non-comparability

`HISTORICAL_RETROSPECTIVE_P4_REFERENCE_NOT_HELD_OUT_NOT_DIRECTLY_COMPARABLE`.
The historical 0.5771428571 Accuracy / 0.5835693346 Macro-F1 result remains
non-held-out, is not paired with the canonical Strict-DG source-only result
(0.1566 / 0.1122), and is not used to estimate the causal effect of target
assistance in this branch. This matched branch does not make the historical
0.1566 and 0.5771 directly comparable — see
`14_HISTORICAL_REFERENCE_AND_NON_COMPARABILITY.md`.

## Limitations of the underlying data

- No independent ground truth exists for the raw P4 CSVs; this branch cannot
  distinguish "the instrument genuinely produced identical repeated readings"
  from "rows were duplicated during dataset assembly." Either way, the
  statistical consequence for the split design is the same.
- No acquisition metadata (session, burst, timestamp, or repeat column) exists
  in the raw CSVs, so grouped holdout by true acquisition unit was not
  possible; leave-condition-cell-out is the closest available proxy.
- Determinism is platform-specific: bitwise re-execution was confirmed on
  Windows CPU with the locked environment; reproduction on another BLAS or
  platform may differ in the last bits.
