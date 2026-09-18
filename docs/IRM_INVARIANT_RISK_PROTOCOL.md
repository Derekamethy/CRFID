# IRMv1 invariant-risk preregistered protocol

This isolated benchmark compares a newly trained matched C1 ERM control with IRMv1. Source measurement positions P1, P2, and P3 are the only environments. P4 is excluded from training, validation, checkpoint selection, annealing, lambda selection, and ranking.

The frozen nonzero grid is `[1, 10, 100, 1000]`. The scalar-logit penalty uses a float32 scale initialized to exactly 1.0, `create_graph=True`, squared per-environment scale gradients, and a mean across active environments. The first `212` canonical optimizer steps use ERM; thereafter the objective is `(mean risk + lambda * penalty) / (1 + lambda)`. The optimizer is not reset at the boundary.

Lambda selection is the specified source-only lexicographic rule. Final P4 predictions from five ERM and five selected-IRM checkpoints must be frozen, checkpoint-bound, hashed, and atomically receipted before P4 TagID or ER is opened. Primary P4 inference uses 63 TagID × ER × surface blocks, not 3,150 rows, with 10,000 paired TagID-stratified bootstrap replicates.
