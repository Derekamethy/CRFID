# Source-internal selection-regret protocol

This isolated study evaluates three and only three source-internal selection events. HOLD_P1 selects with P2+P3 and evaluates P1; HOLD_P2 selects with P1+P3 and evaluates P2; HOLD_P3 selects with P1+P2 and evaluates P3. The candidates are exactly C0 through C3 and the canonical seeds are exactly 42 through 46.

The selector was frozen before numeric `per_run_metrics.csv` access. It is the exact historical lexicographic selector restricted to the two allowed source positions: maximize the worst allowed-position Macro-F1, maximize mean Macro-F1, maximize worst class recall, minimize zero-recall frequency, maximize condition-block Macro-F1, maximize unique-signal-weighted Macro-F1, minimize across-seed population standard deviation, then use ascending candidate ID as the complete tie-break. No normalization or scalar weights are used. The restriction changes only the number of allowed positions from three to two.

The compact released run table does not expose per-class recall vectors. Therefore the third selector component is not silently replaced: execution is valid only when earlier released components strictly resolve every comparison that would otherwise reach it. An unresolved comparison exits as `BLOCKED_NONDECOMPOSABLE_SELECTION_RULE` before held-position evaluation.

Each event first computes allowed-position scores and freezes a canonical SHA-256 decision record. All three decision records are frozen before any held-position accessor is released. Held rows cannot alter ranking, thresholds, tie handling, candidate eligibility, or interpretation. Primary event regret is the held-position oracle candidate's five-seed mean Macro-F1 minus the selected candidate's corresponding five-seed mean. Seed-level paired regrets are also reported. The only primary zero tolerance is `1e-12`, used solely for floating-point nonnegativity and exact-zero checks; the fixed secondary within-tolerance diagnostic is 0.01.

The seed-paired bootstrap uses 10,000 deterministic replicates. Each replicate resamples the five seed identities with replacement and applies the same sampled identities jointly to all candidates, positions, selection scores, and held evaluations. No candidate-position row is treated as an independent bootstrap unit. Leave-one-seed-out analysis and exhaustive symmetric sign perturbations at ±0.001, ±0.0025, and ±0.005 Macro-F1-equivalent are deterministic sensitivity analyses, not probability models.

Any path classified as P4 or target scientific evidence is rejected before opening. The denylist includes P4, target, matched-target, target-evaluation, mixed strict-DG final-result packages, Few-Shot, OpenEMS, and ZIP paths. Full-authority fingerprints are byte-level integrity operations only and are never parsed as scientific inputs.

Selection regret is conditional on the current candidate set and cannot prove regret relative to all possible models or representations.
