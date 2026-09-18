# GroupDRO worst-source-position results

The source-only selector chose **eta = 0.05**. Relative to matched ERM, the source worst-position Macro-F1 contrast was **+0.024357**, 95% CI **[-0.011581, 0.054235]**.

On held P4, the matched block-level Macro-F1 effect was **+0.001405**:

- Block-only 95% interval: **[-0.02435, 0.02510]**.
- Block-plus-seed sensitivity interval: **[-0.04298, 0.03949]**.

The block-level Accuracy effect was **0.000000**; its block-only interval was **[-0.028571, 0.028571]** and block-plus-seed interval **[-0.050794, 0.047619]**. Neither uncertainty scheme supports reliable benefit or harm. Headline cross-method comparisons use the block-plus-seed interval.

The comparison uses unchanged, hash-bound frozen predictions. See the [canonical contrast table](../results/canonical_metrics/groupdro_worst_source/19_P4_PRIMARY_GROUPDRO_VS_ERM_CONTRAST.csv).
