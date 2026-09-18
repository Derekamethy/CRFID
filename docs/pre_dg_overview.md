# Pre-DG Baseline Overview

The authoritative scope contains two separate single-dataset baselines:

| Dataset | Rows | Classes | Evaluations |
| --- | ---: | ---: | --- |
| Paper3 Tyndall | 9,600 | 8 (`0` through `7`) | grouped-random, four leave-position, five leave-surface |
| Paper4 depolarizing | 12,600 | 7 (raw `1` through `7`, model `0` through `6`) | grouped-random, four leave-position, three leave-surface-case |

Every split uses the frozen raw-condition group assignment. Train,
validation, and test raw-condition groups are disjoint. The workflow executes
seeds 42, 43, and 44 and compares dropout candidates 0.0 and 0.3 using
validation data only.

This branch establishes that the data/model pipeline can learn under the
historical non-strict protocol and provides a reference before strict
source-only DG. Grouped-random splits may mix familiar positions or conditions
across partitions. Leave-domain splits are diagnostics under the historical
single-dataset design; they are not the frozen Strict-DG experiment.

See [the protocol](pre_dg_protocol.md) for the exact representation, model,
training rule, and selection rule.
