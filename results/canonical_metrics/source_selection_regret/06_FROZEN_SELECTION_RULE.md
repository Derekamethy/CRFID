# Frozen selection rule

Status: `EXACT_HISTORICAL_SELECTOR_RESTRICTED_TO_TWO_ALLOWED_SOURCE_POSITIONS`.

1. `worst_outer_held_position_macro_f1` — maximize.
2. `mean_outer_held_position_macro_f1` — maximize.
3. `worst_class_recall` — maximize.
4. `zero_recall_class_frequency` — minimize.
5. `condition_block_macro_f1` — maximize.
6. `unique_signal_weighted_macro_f1` — maximize.
7. `across_seed_standard_deviation` — minimize.
8. `candidate_id` — ascending_deterministic.

No scalar weights or normalization are used. Each event replaces the historical three-position aggregation domain with its two allowed positions and changes nothing else. All three decisions are frozen and SHA-256 hashed before any held accessor is enabled.
