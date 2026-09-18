# GroupDRO worst-source-position protocol

This study compares a retrained matched `C1_FIRST_DIFFERENCE_ERM_1DCNN`
control with source-position GroupDRO. The only primary modelling difference is
the risk aggregation rule: ERM uses pooled cross-entropy; GroupDRO computes a
mean cross-entropy for each source position in each batch, updates detached
exponentiated-gradient weights, then backpropagates their weighted sum.

The source groups are exactly P1, P2, and P3. P4 is excluded from all training,
source validation, source LOPO development, eta selection, and checkpoint
selection. The fixed eta grid is 0.01, 0.05, 0.10, and 0.20. Five canonical
seeds are evaluated in the three canonical source LOPO folds.

Both arms use the same deterministic source-position-balanced, TagID-interleaved
batch planner. A position with fewer available samples is repeated only after
all of its original rows have appeared; the policy and incomplete-final-batch
handling are recorded for both arms.

P4 uses a physically separated sealed input view: `p4_signals_float64.npy` and
`p4_unlabelled_registry.csv` may be opened before prediction freeze, while
`p4_labels_int64.npy` cannot be opened until all ERM and GroupDRO predictions
are written and hashed. The primary P4 unit is the 50-row TagID × ER × surface
condition block, reduced by deterministic majority vote with the frozen class
order used to break ties.

Public result tables contain compact metrics, q trajectories, hashes, and
confusions only. Raw measurements, model state, embeddings, logits, and
row-level predictions remain under the ignored `outputs/` runtime boundary.

Interpretation uses the executed source-only selector and matched ERM control
recorded in `results/canonical_metrics/groupdro_worst_source/`. Source-retention,
dispersion, uncertainty, source-evaluation, and epoch definitions are read from
the persisted result tables for this run rather than from a prospective
replication specification.
