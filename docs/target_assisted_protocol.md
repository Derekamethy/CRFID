# P4 Target-Assisted Adaptation protocol

The target-access contract permits P4 features, P4 labels, transductive query features, query-label scoring, and full-P4 metrics for retrospective selection. Adaptation, selection, and evaluation all name `p4_full`; overlap is intentional and reported as a scientific limitation.

The retained readout is `L2(class_mean(L2(source_embedding)))` with float64 cosine scores and lowest-index `argmax` ties. Only source embeddings and source labels create prototypes. Target prototypes, model updates, layer updates, optimizer steps, preprocessing refits, and target-derived hyperparameters are absent.

Claim type: `RETROSPECTIVE_P4_INFORMED_REFERENCE`.

Exactly one run exists (seed 42). No seed sweep, split sweep or repetition was
performed, so the branch carries no stability estimate. The population standard
deviation reported as `0.0` in `aggregate_metrics.json` is the arithmetic
consequence of n = 1 and must not be read as precision or robustness.
