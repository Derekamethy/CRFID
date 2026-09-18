# Pre-DG Results and Artifacts

The complete machine-readable result chain is under `outputs/pre_dg`:

- `custody/`: verified dataset, manifest, schema, class/condition, and split
  identity;
- `protocol_freeze/`: canonical protocol, hash, code/config/split manifests,
  environment, and freeze report;
- `execution/`: all candidate checkpoints and histories, selected predictions,
  per-run metrics, recalls, confusion matrices, and seed aggregates;
- `comparison/`: frozen historical reference, preregistered comparison policy,
  and reproduced-versus-authoritative table;
- `final/`: verdict, final metrics, public report, public manifest, and
  large-artifact omission manifest.

`execution/per_run_metrics.csv` is the source for per-seed results.
`execution/aggregate_metrics.json` is the source for aggregated Accuracy and
Macro-F1. `comparison/REPRODUCTION_COMPARISON.md` records tolerance outcomes.
`final/PRE_DG_BASELINE_REPORT.md` is the concise reviewer-facing result.

Historical predictions and checkpoints were not retained in the frozen
evidence, so their binary identities are not expected to match. The workflow
instead performs model-retraining reproduction and compares metrics against
the preregistered tolerances.

The completed execution verdict is
`FAIL_PRE_DG_RESULT_REPRODUCTION_MISMATCH`. Exact identity checks pass and 53
of 54 per-run Accuracy comparisons are within the `0.05` tolerance. Paper3
leave-surface BEND2 seed 44 is the sole failure: reproduced `0.4984375`,
authoritative `0.5494791666666666`, absolute difference
`0.051041666666666596`. All 18 split means, including both grouped-random
headline means, remain within their respective tolerances.
