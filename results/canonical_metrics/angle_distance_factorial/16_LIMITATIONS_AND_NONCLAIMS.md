# Limitations and nonclaims

- The benchmark contains 63 physical condition blocks and five training seeds;
  uncertainty is therefore block-clustered and not based on 3,150 independent rows.
- The four positions are observational measurement settings. Contrasts are
  angle-associated and distance-associated classification differences, not
  causal physical-mechanism estimates.
- Frozen checkpoint inference reconstructs predictions exactly, but full row
  prediction arrays, logits, signals, and checkpoints are not committed here.
- Macro-F1 is nonlinear. The primary Macro-F1 estimand pools blocks within each
  position and seed; the inherited 15-run mean is descriptive and distinct.
- TagID, ER, surface, seed, fold, per-class, and simple-effect results are
  prespecified secondary or exploratory summaries and are not multiplicity-based
  mechanism discovery.
- Bootstrap intervals describe the available block/seed design and do not add
  new environments, devices, tags, distances, angles, or acquisition sessions.
- The inherited P4 frozen canonical bundle did not contain raw signals; the
  original benchmark's governed raw-file audit and source fingerprint remain the
  controlling P4 data evidence.
- Passed tiny-set learning validity establishes only basic learner capacity on a
  tiny true-label subset, not held-condition generalisation.
- No p-values are reported and no repeated row is treated as an independent unit.
