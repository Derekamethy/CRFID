# P4 Encoder Fine-Tuning Diagnostic v1

Final classification: `ENCODER_ADAPTATION_DOES_NOT_RESCUE_HELD_CONDITION_TRANSFER`

This isolated target-assisted diagnostic compares the completed frozen-linear parent with Partial and Full C1 representation adaptation under identical block-disjoint P4 supports and queries.

| Budget | Frozen F1 | Partial F1 | Partial-Frozen | Full F1 | Full-Frozen | Full-Partial |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.1267 | 0.1251 | -0.0016 | 0.1248 | -0.0019 | -0.0003 |
| 35 | 0.1447 | 0.1403 | -0.0044 | 0.1361 | -0.0086 | -0.0042 |
| 100 | 0.1443 | 0.1360 | -0.0083 | 0.1330 | -0.0113 | -0.0030 |
| 500 | 0.1427 | 0.1319 | -0.0109 | 0.1296 | -0.0131 | -0.0022 |

All new predictions were persisted and hash-registered before P4 query labels were opened. Hyperparameters were selected source-only and frozen across budgets.

Reproducibility: historical target-labelled implementation requiring omitted hash-bound parent assets and governed P4 inputs; see [reproducibility](../../../REPRODUCIBILITY.md).
