# Unified source-only DG benchmark

This compact result set compares **within-branch matched effects** for IRM, DANN v2 and GroupDRO on held P4. Absolute scores from different branches are not ranked because their samplers, controls and estimands differ.

| Method | Matched P4 Macro-F1 effect | 95% interval | Conclusion |
|---|---:|---|---|
| IRM | -0.019330 | [-0.066939, 0.025387] | No reliable benefit |
| DANN v2 | +0.016037 | [-0.035578, 0.068624] | No reliable benefit |
| GroupDRO | +0.001405 | [-0.042978, 0.039489] | No reliable benefit |

Domain-aware Mixup is excluded from performance comparison because its frozen protocol reached only 50% lawful-parent coverage and stopped before training or P4 evaluation.

The experiment does not establish reliable position suppression from DANN. Its position-probe change was weak and uncertain, and the causal role of position encoding remains unresolved.

Files:
- `matched_effects.csv`: compact branch-level benchmark data.
- `matched_effects_forest_plot.png`: matched-effect visualisation.

These are source-only DG results. Target-labelled, retrospective and within-condition target-fitting experiments answer different questions and are intentionally excluded.