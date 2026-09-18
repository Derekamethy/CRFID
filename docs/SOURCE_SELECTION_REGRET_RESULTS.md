# Source-internal selection-regret result

Classification: `SOURCE_SELECTION_VALIDITY_PARTIALLY_SUPPORTED`

- Evidence complete: yes, 60/60 candidate × held-position × seed rows, each independently replayed from a hash-bound source prediction bundle.
- Reruns: none; execution case A used frozen source-only evidence.
- Selector: exact historical lexicographic selector restricted to the two allowed source positions.
- Selections: HOLD_P1 -> C1_FIRST_DIFFERENCE_ERM_1DCNN; HOLD_P2 -> C1_FIRST_DIFFERENCE_ERM_1DCNN; HOLD_P3 -> C1_FIRST_DIFFERENCE_ERM_1DCNN.
- Held-position oracles: HOLD_P1 -> C1_FIRST_DIFFERENCE_ERM_1DCNN; HOLD_P2 -> C3_SOURCE_CORAL_ERM_1DCNN; HOLD_P3 -> C1_FIRST_DIFFERENCE_ERM_1DCNN.
- Mean/median/maximum three-event regret: 0.0134394211597 / 0 / 0.040318263479.
- Top-1 recovery: 2/3; zero-regret events: 2/3.
- Seed stability: see `13_SEED_WINNER_STABILITY.csv`; the bootstrap used paired seed units only.
- Simpler baseline with lower mean regret: none (canonical mean regret 0.0134394211597).
- P4 seal: intact; the workflow's synthetic P4 access was rejected before open.

Selection regret is conditional on the current candidate set and cannot prove regret relative to all possible models or representations.
