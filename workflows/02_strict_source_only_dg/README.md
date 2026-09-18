# Strict source-only DG

P1-P3 determine preprocessing, candidate and checkpoint selection. P4 is reserved for final evaluation. The selected candidate is C1 first-difference ERM.

## Public compact-evidence verification

From the repository root, in the installed public environment:

```powershell
./.venv/Scripts/python workflows/02_strict_source_only_dg/run.py --config configs/strict_dg/canonical.yaml --execute
```

This read-only command validates [retained public evidence](../../results/canonical_metrics/strict_dg/): source ranking, metric tables, five checkpoints, preprocessing and confusion matrices. Success is `PUBLIC_COMPACT_EVIDENCE_VERIFIED`. Without `--execute`, it returns a plan. It does not train, read measurements, issue target access, or replay omitted predictions.

## Historical execution

The staged source-selection, training and P4-inference implementations remain under `scripts/`. Full retraining/replay requires governed measurements and omitted source/split, recipe and historical prediction/release inputs. Measurements alone are insufficient. Historical full-release replay is not distributed.

See [reproducibility](../../REPRODUCIBILITY.md), [protocol](../../docs/strict_dg_protocol.md), and [results](../../docs/strict_dg_results.md). Source-only access rules remain distinct from later target-labelled diagnostics.
