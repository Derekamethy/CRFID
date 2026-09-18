# Data lineage

## Governed measurement authority

The primary Tyndall/Paper4 dataset contains 12,600 rows:

```text
7 TagIDs x 3 ER conditions x 4 positions x 3 surfaces x 50 repetitions
```

Each position therefore contains 3,150 rows and 63 complete condition blocks.
P1-P3 contain 9,450 source rows; P4 contains 3,150 target rows. The independent
split audit confirms zero P4 rows in the Strict-DG source registry and no
sample, exact-signal, or condition-block overlap across source train,
validation, and held-position partitions.

Raw measurements are external. The sealed Strict-DG package records the P4
file fingerprints:

| External file | Role | SHA-256 |
|---|---|---|
| `A1_P4.csv` | P4 surface A1 | `8f63f8bee3deb7747a88ac0fe1f47eabe8c0feedfe5a65cb503e25fdcf1377ee` |
| `A2_P4.csv` | P4 surface A2 | `6b20fb815d6df40ff9ad4c0ecbf3bc7a82762ac96b8fd7df7170504715d75a00` |
| `A3_P4.csv` | P4 surface A3 | `6c3bffcca0e81664398423c5c4828c3d7d4698a69e804fe69368e53bb8a9f6e3` |

The public schemas are in `data/schemas/datasets.json`. Raw TagID labels 1-7
are mapped explicitly to public classes 0-6. Data1 has four local labels and
no assumed class mapping to the seven Tyndall TagIDs.

## Strict-DG chain

```text
external P1-P3 CSV measurements
  -> governed 281-value spectrum parsing and label mapping
  -> source-only registry (9,450 rows; SHA-256 a1cbb062...)
  -> source-only LOPO/grouped split (SHA-256 41a275a6...)
  -> source-fitted preprocessing per governed recipe
  -> candidates C0-C3 evaluated only on P1-P3 source folds
  -> C1 selected and recipe/checkpoints sealed (seeds 42-46)
  -> P4 CSVs opened once for final inference
  -> five immutable P4 prediction NPZ files
  -> per-seed accuracy, Macro-F1, confusion matrices
  -> FINAL_P4_RESULTS.csv/json
  -> unified benchmark and scientific summaries
```

The compact public evidence is under
`results/canonical_metrics/strict_dg/`. It retains the final metrics,
source-selection summaries, confusion matrices, preprocessing state, and five
selected checkpoints. Source signal/label arrays, full split CSVs, and large
prediction/logit bundles are not distributed in the portfolio copy.

No P4 information affects Strict-DG normalisation, feature construction,
candidate definition, hyperparameters, architecture, seed choice, model or
checkpoint selection. P4 labels/features affect final evaluation only.

## Expanded source-only DG methods

IRM, DANN v2, GroupDRO, and Mixup share the source-only C1 data contract
but retain method-specific protocols and controls:

```text
governed P1-P3 source inputs
  -> method-specific source split/config
  -> source-only hyperparameter/epoch selection
  -> pre-P4 seal/access record
  -> method + branch-local ERM prediction freeze
  -> P4 block-majority metrics and paired effect interval
  -> compact canonical result directory
```

IRM, DANN v2, and GroupDRO reached final P4 evaluation. Mixup stopped before
training because the independently generated frozen split supplied only 21 of
42 lawful shared keys (2,100/4,200 required parents). Its result tables are
schemas/zero-run evidence, not model performance.

GroupDRO's P4 metrics were rescored from unchanged, hash-bound frozen
predictions after an output-persistence interruption. Model weights, predictions,
labels, estimand, and bootstrap definition did not change. The compact final
metrics and the corresponding limitation note are retained directly in
`results/canonical_metrics/groupdro_worst_source/`.

## Target-labelled and retrospective chains

Few-shot, factor-aware few-shot, large calibration, trainable readout, encoder
fine-tuning, retrospective P4, and matched target-assisted P4 legitimately use
P4 information.

```text
frozen source carrier/checkpoint
  + labelled P4 support or full-P4 target information
  -> adaptation/readout/selection
  -> held-query or retrospective P4 prediction
  -> target-labelled metric and bounded conclusion
```

These chains never feed the canonical Strict-DG C1 selection or result. The
historical 0.577/0.584 result used full-P4 outcomes in readout selection and is
therefore retrospective. Labelled-target outputs are isolated in their own
configs, workflows, results, and registry labels.

## Representation and physical diagnostics

Representation probes, fixed-position grouped validation, peak missingness,
angle-distance contrasts, and source-selection regret reuse frozen
measurements/predictions for diagnosis. They may inspect P4 labels/features,
but do not alter the Strict-DG checkpoint or claim. Their lineage is:

```text
frozen measurement/prediction/embedding identity
  -> preregistered or fixed diagnostic transformation
  -> block-/seed-aware contrasts and intervals
  -> diagnostic-only interpretation
```

OpenEMS is separate simulation evidence:

```text
measured reference/diagnostic observation
  -> canonical geometry/config assumptions
  -> historical baseline and exploratory redesign simulation outputs
  -> confirmation/gate analysis
  -> NOT_CONFIRMED interpretation
```

Simulation does not feed a validated hardware measurement or classification
claim.

## External Data1

Data1 A is a within-Data1 portability study. Data1 SSL B uses external Data1 in
a governed auxiliary treatment and opens Tyndall P4 only after its gate. Data1
labels 0-3 are local; no mapping to Tyndall classes 0-6 is inferred. Neither
branch is a replacement for the canonical Strict-DG data chain.

## Interpretation safeguards

Current public claims do not treat historical random splits, target-assisted
readout promotion, or cross-lineage absolute scores as substitutes for the
Strict-DG result. Each result directory states its own information-access and
comparability boundary.
