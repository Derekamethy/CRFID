# Limitations and nonclaims

- This is a diagnostic representation comparison, not model selection, architecture search, or a causal RF-mechanism experiment.
- Angle results are observational and are described only as angle-associated and representation-dependent.
- The frozen C1 encoder necessarily uses its released P1-P3 source preprocessing state. That state is checkpoint-bound, not refitted to each diagnostic fold; the subsequent probe scaler is training-only.
- C1 checkpoint seeds and probe seeds are deliberately aligned. Their variance is therefore an aligned checkpoint-plus-optimization quantity and cannot identify the two sources independently.
- Only one fixed linear probe and one deterministic cosine-NCM secondary readout were used. No conclusion extends to nonlinear readouts or retuned encoders.
- The peak implementation and windows were inherited, not optimized. Locations are reported primarily as ordered indices because the exact physical frequency vector is only conditionally supported.
- The inferential sample is 63 condition blocks per position. The 50 repeated rows within a block are descriptive replicates and are never treated as independent bootstrap units.
- Percentile bootstrap intervals quantify finite-block sampling under TagID stratification; they do not cover every source of measurement, checkpoint, or design uncertainty.
- Validation and test each contain 21 blocks per fold. Early stopping may remain variable despite the five fixed seeds.
- Results concern these seven TagIDs, three ER levels, three surfaces, and four measured positions only; no external-domain or OpenEMS generalization is claimed.
- Test labels were opened once per frozen prediction vector. No post-result tuning was performed.
- No raw signal, processed array, checkpoint, embedding, logit, row-level prediction, private sample identifier, or absolute local path is distributed in this diagnostic evidence directory.
