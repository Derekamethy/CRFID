# P4 Target-Assisted Adaptation

This workflow reproduces the retained retrospective P4-informed source cosine NCM readout. It verifies the immutable strict source release, loads the separately identified historical frozen embedding carrier, fits prototypes from 7,350 source rows only, predicts all 3,150 P4 rows, and uses P4 labels for readout-selection reconstruction and final metrics.

The full P4 partition is reused for selection evidence and reporting. There is no independent P4 query set, no target prototype, no optimizer, and no model update. The only allowed claim is `RETROSPECTIVE_P4_INFORMED_REFERENCE`.

Set `CRFID_TYNDALL_P4_CSV` and `CRFID_P4_EMBEDDING_BUNDLE`, then run the locked interpreter:

```powershell
crfid-python.cmd workflows/03_target_assisted_adaptation/run.py --config configs/target_assisted/canonical.yaml --mode full
```

Narrow modes are: `validate_source_release`, `validate_target_inputs`, `build_target_partitions`, `prepare_target_assisted_method`, `execute_target_adaptation`, `generate_target_predictions`, `calculate_target_metrics`, `aggregate_target_results`, `reproduce_historical_selection`, `compare_with_authoritative_outputs`, and `generate_target_assisted_report`.
