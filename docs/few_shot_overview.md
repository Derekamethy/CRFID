# Few-shot P4 adaptation overview

This study measures how a small amount of labelled P4 support changes performance relative to the frozen source-only carrier. It is a target-domain adaptation experiment and is intentionally separate from Strict-DG.

The historical episode-level prototype Macro-F1 values are approximately:
- zero-shot: **0.1073**
- 1-shot: **0.1122**
- 3-shot: **0.1451**
- 5-shot: **0.1444**

Support and query roles are separated within each episode, and query labels are opened only after predictions are fixed.These values do not establish independent-campaign generalization because all episodes reuse one P4 measurement corpus. They quantify within-corpus calibration behaviour only.

Public components:
- `configs/few_shot/canonical.yaml` — protocol declaration
- `src/crfid/few_shot/` — support/query and result utilities
- `workflows/05_few_shot_p4_adaptation/` — workflow entry point
- `results/canonical_metrics/few_shot/` — compact result summary

See `few_shot_protocol.md` and `few_shot_claims.md` for the access and interpretation boundaries.