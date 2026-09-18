# IRMv1 invariant-risk executive summary

Governed runtime binding succeeded: `True`. IRMv1 implementation validity passed: `True`. Second-order penalty gradients reached the encoder: `True`. Focused test suite: `63 passed`.

Development runs: `75`. Final runs: `10`. Selected lambda: `1`. Annealing point: optimizer step `212` of `850` canonical steps.

Source mean Macro-F1 contrast: `0.013903` (95% `[-0.014233215052822988, 0.050651785012082395]`). Source worst-position Macro-F1 contrast: `0.039410` (95% `[-0.022333281478899597, 0.06072708355522666]`). Source retention passed: `True`. Source mean improved: `True`. Source worst position improved: `True`.

Validation IRM penalty contrast: `-0.039185` (95% `[-0.10834783247516801, 0.030150150548555146]`); decreased reliably: `False`. Environment-risk dispersion contrast: `0.014487` (95% `[0.00518769812459747, 0.022851284729937714]`); decreased reliably: `False`. Position decodability contrast: `-0.025397` (95% `[-0.09206349206349207, 0.012698412698412653]`).

P4 block Macro-F1 contrast: `-0.019330` (block-only `[-0.055264673240522234, 0.01875015663035029]`, block-plus-seed `[-0.06693913485322048, 0.025387052020226227]`); improved: `False`. P4 block Accuracy contrast: `0.025397` (block-only `[-0.015873015873015872, 0.06349206349206349]`, block-plus-seed `[-0.025396825396825397, 0.07619047619047618]`); improved: `True`. Effects survived both block and seed uncertainty: `False`.

Invariance-diagnostic classification: `SOURCE_INVARIANCE_DIAGNOSTICS_NOT_IMPROVED`. Main scientific classification: `IRM_MIXED_OR_UNSTABLE_RESULT`.

Predictions were frozen and hashed before P4 labels were opened. Secondary reporting reused already-persisted diagnostics without reopening labels or recomputing primary metrics.
