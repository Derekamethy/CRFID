# DANN position-suppression intervention protocol

This isolated intervention compares a matched C1 first-difference ERM control with a C1 model carrying one gradient-reversed source-position head. P1–P3 are the only development and training domains. P4 is a final held target and cannot influence epochs, checkpoints, lambda selection, probes, or diagnostics.

The C1 encoder is the canonical 280-point first-difference 1D CNN through its 256-dimensional embedding. The TagID head, AdamW optimizer, maximum 50 epochs, validation Macro-F1 checkpoint rule, patience eight, five seeds, and S1–S3 folds are bound to the canonical release. Both methods use the same deterministic position-by-TagID interleaved sampler.

The domain head is fixed at 256 → 128 → K with ReLU and zero dropout. K is two in source LOPO development and three in final P1–P3 training. Its loss is ordinary cross-entropy. The encoder receives that loss only through the frozen reversal schedule `lambda_max * (2 / (1 + exp(-10p)) - 1)`. The grid is exactly 0, 0.03, 0.10, 0.30, and 1.00.

One nonzero lambda is selected lexicographically from source-only held-position TagID Macro-F1 and an independent frozen-encoder position probe. P4 features are loaded without exposing TagID or ER. All ten P4 prediction arrays are serialized and hashed before a second, gated scoring stage opens labels once.

The primary target endpoint is 63-block Macro-F1 using deterministic row-prediction majority vote with the lowest global class index as the tie break. Paired TagID-stratified block bootstraps use 10,000 deterministic replicates. Repeated rows are never inferential units.

This intervention can strengthen mechanism evidence but is not formal causal mediation and cannot establish definitive physical causality.
