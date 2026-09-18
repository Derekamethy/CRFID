# Paired 2x2 angle-distance factorial contrast protocol

This file records the preregistered protocol before final factorial estimates
were computed. The controlling full protocol is
`docs/ANGLE_DISTANCE_FACTORIAL_PROTOCOL.md`; its frozen choices are summarized
here for the results bundle.

- Base: fixed-position commit `4187fc44e05804ae1ad9ebbdf889ad8ca6f9e6d4`.
- Cells: P1 = 50 mm/0 degrees, P2 = 50 mm/45 degrees,
  P3 = 150 mm/0 degrees, P4 = 150 mm/45 degrees.
- Primary unit: 63 paired physical `TagID x ER x surface` blocks.
- Primary endpoint: condition-block accuracy, averaged across five seeds.
- Angle: `0.5 * ((P2 - P1) + (P4 - P3))`.
- Distance: `0.5 * ((P3 - P1) + (P4 - P2))`.
- Interaction: `(P4 - P3) - (P2 - P1)`.
- Primary uncertainty: 10,000 paired TagID-stratified block bootstrap
  replicates, seeds fixed, random seed 20260804.
- Sensitivity: the same paired block bootstrap with paired seed resampling.
- Macro-F1: pool 63 aligned blocks within position and seed, compute Macro-F1,
  average five seed-specific values, then contrast; do not invent block Macro-F1.
- Rows: descriptive only; never independent inferential units.
- P-values: not reported.
- Audit decision: Case B. The inherited test folds are synchronized, but 492
  train/validation assignment groups differ because the inherited validation RNG
  included position. The inherited 60 runs are not used for factorial inference.
- Synchronized rerun: exactly 60 C1 primary runs, three unchanged outer folds,
  seeds 42-46, no tuning, no other model, and a position-independent validation
  RNG keyed by protocol, fold, seed, and TagID. This amendment was frozen before
  synchronized training began.
- Learning validity: reuse the completed four-position tiny-set evidence without
  rerunning or tuning it.
