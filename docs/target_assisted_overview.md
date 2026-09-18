# P4 Target-Assisted Adaptation overview

This branch reproduces a retrospective target-informed readout result. It replaces the historical learned head with source-only normalized class prototypes and scores all P4 embeddings by cosine similarity. P4 outcomes influenced the retained readout choice, so the result is not an unseen-target or prospective evaluation.

## What this branch is not

It is not strict source-only domain generalisation, not an unseen-target evaluation,
not a multi-seed robustness result, and not evidence that target assistance alone
caused the gap against the strict source-only figure. It is a single-lineage,
single-seed reference point obtained under disclosed full-P4 outcome access.
