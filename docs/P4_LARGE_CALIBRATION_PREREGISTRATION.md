# P4 Larger Labelled-Target Calibration Preregistration

The normative preregistration is
`configs/p4_large_calibration_curve/preregistration.json`. It was sealed before
any new calibration-curve model result or query score was computed.

This is a target-assisted labelled-target calibration study, not strict domain
generalisation at any positive budget. It holds the five source-only C1 encoders,
P1-P3 preprocessing, cosine-prototype rule, query folds, seeds, and analysis
fixed while varying only the number of labelled P4 support measurements.

The total-label budgets are 0, 7 (1-shot), 10, 20, 21 (3-shot), 35 (5-shot),
50, 100, 200, and 500. A second axis records independent physical condition
blocks. The fixed three-fold out-of-fold query always contains all 3,150 rows
and all 63 blocks exactly once. Within a fold, support can cover at most 42
independent blocks; labels beyond 42 are repeated acquisitions.

Supports are deterministic, nested, TagID-stratified, near-balanced, and row-
disjoint. The sampler covers every eligible class-by-condition block before
reusing a block. Twenty support seeds are crossed with the five frozen source
seeds. All predictions are generated and hash-registered before any query-label
seal opens.

The primary endpoint is full-P4 out-of-fold sample Macro-F1, equally averaged
over crossed source/support units. Uncertainty is a 10,000-replicate paired
hierarchical percentile bootstrap that resamples physical blocks within TagID
and independently resamples source and support seeds. Rows are never bootstrap
units. The key threshold is the first budget whose Macro-F1 improvement is at
least 0.05 and whose paired 95% interval excludes zero.

The historical retrospective and matched within-condition target-assisted
figures are not protocol-comparable endpoints, so recovered-gap percentages are
preregistered as not identifiable. No result-dependent method, budget, seed,
threshold, or uncertainty change is permitted.
