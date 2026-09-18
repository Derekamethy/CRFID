# P4 factor-aware few-shot selection

This workflow compares matched-budget random and factor-aware P4 support selection under three condition-block-disjoint Latin-square query folds. It binds the frozen C1 source models, seals query labels until predictions are frozen, and uses physical condition blocks for evaluation.

Prerequisite-only validation:

```powershell
python workflows/11_p4_factor_aware_few_shot/run.py `
  --archive-repository <frozen-source-archive> `
  --data-directory <governed-p4-directory> `
  --output-directory outputs/p4_factor_aware_few_shot_dry_run `
  --dry-run
```Omit `--dry-run` for the registered scientific execution. Public result tables omit raw signals, embeddings, logits, checkpoints and row-level predictions.