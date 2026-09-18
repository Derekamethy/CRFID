# Pre-DG Baseline

This workflow executes the final Stage1-5 Paper3 and Paper4 single-dataset
baselines over the authoritative grouped-random and leave-domain splits. It
hash-verifies the processed inputs and splits, fits preprocessing on train only,
trains both frozen dropout candidates for seeds 42/43/44, selects by validation
only, saves and reloads checkpoints, generates predictions, recomputes metrics,
and compares selected runs with the frozen historical evidence.

The complete execution currently has one per-run Accuracy mismatch beyond the
preregistered tolerance; see `outputs/pre_dg/final/final_verdict.json`.

This is a non-strict learnability baseline. It is not strict DG, target
adaptation, external validation, unseen-position proof, or a fair zero-shot DG
benchmark. See `PRE_DG_START_HERE.md`.
