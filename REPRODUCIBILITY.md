# Reproducibility

Raw Tyndall/Paper4 measurements are governed external data and are not downloaded or distributed by this repository. Public availability of code and compact evidence does not imply that every historical execution can be replayed from measurements alone.

## Environment

For public development, tests and retained-evidence verification, use Python >= 3.11:

```powershell
python -m venv .venv
./.venv/Scripts/python -m pip install --upgrade pip
./.venv/Scripts/python -m pip install -e ".[ml,test]"
```

On Linux/macOS use `.venv/bin/python`. The exact historical environment lock is a separate replay contract; it is not required for public tests. See [environment setup](environment/ENVIRONMENT_SETUP.md). OpenEMS/CSXCAD are optional external dependencies.

## A. Publicly reproducible now

```powershell
./.venv/Scripts/python tools/validate_configs.py
./.venv/Scripts/python tools/check_import_safety.py
./.venv/Scripts/python -m pytest -q
./.venv/Scripts/python workflows/02_strict_source_only_dg/run.py --config configs/strict_dg/canonical.yaml --execute
```

The first validator checks YAML configurations, including component fragments. Unit, synthetic and compact-evidence tests run without measurements; tests requiring omitted measurements or historical runtime bundles skip explicitly.

The Strict-DG command returns `PUBLIC_COMPACT_EVIDENCE_VERIFIED`. It checks the retained candidate ranking, final CSV/JSON metrics, five checkpoint hashes, preprocessing state/arrays, and confusion-derived Accuracy/Macro-F1. It accounts explicitly for historical CRLF versus distributed LF text serialization. Recorded input hashes describe identity; they cannot verify absent raw inputs. No measurements are opened, no model is trained, and no target authorization is issued.

The verified five-seed means are Accuracy **0.1566349206** and Macro-F1 **0.1122308836** (seeds 42-46). Full historical prediction/logit replay is `HISTORICAL_FULL_RELEASE_REPLAY_NOT_DISTRIBUTED`.

## B. Execution with governed external measurements

### Paired angle-distance diagnostic

Supply the approved twelve CSVs `A1_P1.csv` through `A3_P4.csv` directly under `--raw-root`. The loader validates the original header, 281 ordered signal values, labels, 50 repeats per condition, 63 condition blocks per position, and retained measurement fingerprints. The general schema is in [datasets.json](data/schemas/datasets.json).

```powershell
./.venv/Scripts/python workflows/10_angle_distance_factorial_contrast/run.py --raw-root '<approved-csv-directory>'
```

This is read-only preflight. Add `--execute --output-root '<new-output-directory>'` only to perform the full synchronized Case-B study. It uses the fixed C1 recipe, four positions, three folds, seeds 42-46, 60 runs, train-only preprocessing, no tuning, and the declared paired block/seed bootstrap. It is a target-labelled physical diagnostic, not Strict-DG selection. No historical Git repository, branch, patch, or exported change list is required. Outputs go under `outputs/` or an external directory, not retained canonical evidence. Historical repository-provenance replay is not distributed. A full training run is substantial and is not part of the lightweight checks.

### Prepared-array validation

`workflows/00_prepare_data/run.py --config configs/data/tyndall.yaml --execute` validates an externally supplied `${CRFID_DATA_ROOT}/tyndall_source_input.npz` containing aligned `signals`, `labels`, `domains`, and `conditions`. It requires 281-value signals and P1-P3 domains; it does not convert raw CSVs into the full historical source registry. Output defaults to `${CRFID_OUTPUT_ROOT:-outputs}/prepared/tyndall.npz`.

## C. Historical and retained implementations

The remaining historical training, adaptation, and replay entry points require branch-specific processed inputs, frozen manifests, checkpoints, predictions or reference bundles in addition to measurements. Their source code and compact results remain inspectable; they are not advertised as raw-measurements-only reproduction commands.

- **Full Strict-DG replay/retraining:** `scripts/reproduce_strict_dg_source_selection.py`, `train_and_freeze_strict_dg.py`, and `evaluate_frozen_p4.py` retain the staged implementation. They require the omitted source registry/splits, 60-unit candidate prediction evidence, selected recipe, release bindings and raw measurements. The public compact verifier replaces none of those inputs and does not claim their reproduction.
- **Encoder fine-tuning:** retained historical/supporting implementation. `--archive-repository` is an external directory containing the original hash-bound `results/canonical_metrics/p4_trainable_linear_readout/` parent files, including `per_run_linear_predictions.npz`, binding/status files and its original artifact manifest, plus calibration support/fold manifests under `results/canonical_metrics/p4_large_calibration_curve/`. It must also contain `outputs/few_shot/frozen_branch/01_frozen_source_import/frozen_last_code_snapshot/`, with `10_final_recipe_freeze/` preprocessing arrays and `11_final_p4_evaluation/runs/seed_42` through `seed_46/FINAL_CHECKPOINT.pt`. A Git object store or particular branch is not required. Sanitized compact files are not substitutes for original files when their historical byte hashes differ.
- Fine-tuning uses governed P4 CSVs via `--data-directory`, the retained source-selected configuration, and the exact inherited support/query plan. Source-selection replay additionally requires the original `outputs/few_shot/frozen_branch/05_head_finetuning/fs4_source_recipe_selection/` protocol/episode material and hash-bound source arrays/registry. Full fine-tuning source-retention diagnostics require `CRFID_SOURCE_SIGNALS` and `CRFID_SOURCE_REGISTRY`. Use a new output directory; the public code never requires restoration of those inputs into Git.

```powershell
./.venv/Scripts/python workflows/14_p4_encoder_finetuning/run.py --dry-run --archive-repository '<external-scientific-archive>' --data-directory '<approved-p4-csv-directory>'
```

Fine-tuning preflight writes nothing and explains missing archive inputs before loading target data. Hash, grouping, support/query, checkpoint, and label-access gates remain active. Other historical workflows, including few-shot and Data1 replay, retain their individual input contracts; compact results can be read without those archives.

## D. Retrospective and target-informed supporting analyses

Target-labelled calibration, representation probes, fine-tuning, the physical factorial diagnostic, historical readout promotion and within-condition target fitting answer different questions from Strict-DG. Their results do not feed C1 selection. Branch-local DG effects use their own matched ERM comparators. OpenEMS remains exploratory simulation evidence with `NOT_CONFIRMED` status and no hardware validation.

## Historical metadata

Tokenized paths and historical identities inside retained configuration/binding JSON are inert recorded metadata, not instructions to reconstruct private repositories. Those evidence files remain unchanged to preserve their scientific record. Some recorded hashes refer to omitted artifacts or original serialization; only checks explicitly reported by a public verifier have been performed. Reader-facing indexes point to retained evidence.
