# Strict source-only DG

P1-P3 were used for fitting, preprocessing, candidate selection and checkpoint selection; P4 was held outside source-side development until final evaluation.

The selected C1 first-difference 1-D CNN reached:
- Accuracy: **0.156635 ± 0.030516**
- Macro-F1: **0.112231 ± 0.018956**
- Five seeds: 42-46
- P4 samples per seed: 3,150

This is the canonical source-only result and a completed negative transfer result, not a failed execution.

This compact public bundle retains final metrics, source-selection summaries, confusion matrices, preprocessing state and the five selected checkpoints. Large prediction/logit bundles used for forensic bitwise replay are intentionally omitted from the public portfolio copy.

The result should not be compared directly with target-labelled, retrospective or within-condition target-fitting experiments because those use different information-access regimes.