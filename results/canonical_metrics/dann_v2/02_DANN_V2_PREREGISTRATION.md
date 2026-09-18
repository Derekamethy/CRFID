# DANN v2 preregistration

Frozen before development aggregate access. Experiment: `POST_HOC_SOURCE_ONLY_FOLLOWUP_DANN_V2`.

The question is whether magnitude-matched, treatment-consistent source-position adversarial training reduces frozen-probe position decodability while retaining TagID information. P4 is permitted only after that source mechanism passes.

Arms are `ARM_A0_ERM_MATCHED`, `ARM_A1_DOMAIN_POSITIVE_MATCHED`, and `ARM_A2_DANN_NEGATIVE_MATCHED`. A1 and A2 use identical domain heads and `lambda(p)` magnitudes; encoder coefficients are respectively positive and negative. A0 receives no encoder-side domain gradient.

The immutable lambda grid is `0.01, 0.03, 0.10, 0.30, 1.00`. Development contains 15 A0 runs, 75 A1 runs, and 75 A2 runs: exactly 165 fold-by-seed treatment runs. This count will not change after outcomes are visible.

For each run, checkpoint selection maximizes familiar-position `inner_validation` row Macro-F1. Frozen probes train on `inner_train` condition-block centroids and evaluate on `inner_validation` centroids. Outer-held LOPO is recorded only as a diagnostic beside `1/7` and never ranks lambda.

Candidate A2 eligibility requires both mean differences versus matched A0: familiar-source validation Macro-F1 at least `-0.02`, and TagID-probe Macro-F1 at least `-0.05`. It also requires mean position-probe difference below zero. A candidate is strong if the paired 95% interval upper bound is below zero. If no strong candidate exists, the preregistered fallback permits `WEAK_POSITION_REDUCTION_CANDIDATES` whose mean is below zero and both retention gates pass despite an interval crossing zero.

Among eligible candidates, rank greatest position-probe reduction first, then higher TagID-probe Macro-F1, then higher familiar-source validation Macro-F1, then smaller lambda. Absolute tie tolerance is `0.002` Macro-F1 at applicable stages.

All development intervals use 10,000 paired fold-by-seed bootstrap replicates with the single seed `20260809`. The final epoch for each seed is the deterministic median of that seed's three selected A2 checkpoint epochs. A0/A1/A2 then receive the identical seed-wise epoch count.

If no candidate achieves selective invariance, classification is `NO_DANN_V2_CANDIDATE_ACHIEVES_SELECTIVE_INVARIANCE`; P4 remains unopened and every P4 result file records `NOT_CREATED_BY_GOVERNED_DESIGN`.

If P4 is permitted, predictions are frozen for 15 models before labels are read. P4 intervals use 10,000 percentile replicates, TagID-condition-block stratification, and the single seed `20260809` for every contrast and family. Chance uses analytic accuracy `1/7` plus 10,000 exact-structure uniform-random block replicates with seed `20260810`.

