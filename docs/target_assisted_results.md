# P4 Target-Assisted Adaptation results

Seed 42 reproduces Accuracy `0.5771428571428572` and Macro-F1 `0.5835693346352661` exactly. Per-class recall is `0.5577777777777778, 0.8333333333333334, 0.6688888888888889, 0.6666666666666666, 0.43333333333333335, 0.43555555555555553, 0.4444444444444444`.

The strict source-only result is `0.15663492063492063 / 0.11223088363457531`, reported as a five-seed mean (42-46). It remains the answer to the strict-DG question; the target-assisted result is a separate reference point under disclosed P4 outcome access.

The two numbers are **not comparable**, and the `~0.421` accuracy difference must not be attributed to target access. On the same historical embedding carrier, a purely source-only readout selector (`whiten_l0.75`) already reaches Accuracy `0.5355555555555556` and Macro-F1 `0.5382807709220131` using no P4 information at all. Roughly 90 % of the difference is therefore present before any P4 outcome is consulted and is attributable to the different source-representation lineage; retrospective P4-informed promotion accounts for only about `0.042` absolute accuracy.

This is a single run at seed 42. The `0.0` dispersion in the aggregate file follows from n = 1 and is not a stability claim.
