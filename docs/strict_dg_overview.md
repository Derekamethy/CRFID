# Strict-DG overview

The Strict-DG study asks whether a TagID classifier developed entirely on P1-P3 can transfer to the held P4 joint geometry.

## Source-only selection

Four candidates were compared on source-domain held-position folds: neutral ERM (C0), first-difference ERM (C1), a source NCM readout (C2), and CORAL ERM (C3). Seeds 42-46 were equally weighted. Candidate selection used only P1-P3 evidence; P4 did not influence preprocessing, hyperparameters, architecture, checkpoint selection, or model selection.

C1 was selected. It differences the 281-point input to length 280, fits featurewise source-only normalization, and applies a 142,855-parameter GroupNorm 1-D CNN.

## Held-P4 result

Across five seeds, C1 reached:
- Accuracy: **0.156635 ± 0.030516**
- Macro-F1: **0.112231 ± 0.018956**

The 3,150 P4 sweeps correspond to 63 physical condition blocks. Repeated sweeps are therefore not interpreted as independent deployment conditions.

The supplied 281-point representation can only be conditionally mapped to approximately 5-8 GHz because the authoritative original frequency vector and exact crop indices are unavailable.

Compact final evidence, the five selected checkpoints, preprocessing state, source-selection summaries, and confusion matrices are in `results/canonical_metrics/strict_dg/`.