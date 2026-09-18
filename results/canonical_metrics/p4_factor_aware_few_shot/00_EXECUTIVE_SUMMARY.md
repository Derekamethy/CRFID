# Executive Summary

Main classification: `NO_RELIABLE_FACTOR_AWARE_COVERAGE_BENEFIT`.

The historical Few-Shot split was row-disjoint and condition-block-disjoint; it
was not condition-block-overlapping. Its 3-shot supports always covered all ER
levels, but full three-surface coverage occurred in only 21 of 126
episode-by-class cases. The new Latin-square protocol is fully
condition-block-disjoint and uses exactly matched 7/21/35-label budgets.

For the preregistered 3-shot cosine-prototype comparison, mean block Macro-F1 was
0.129095 for random support and
0.124103 for joint factor coverage. The paired
joint-minus-random effect was -0.004992
(95% block-bootstrap interval [-0.039200, +0.029010]). The
matched block-Accuracy effect was -0.005714
([-0.041905, +0.030476]).

Primary factor-aware improvement: **no**.
Budget-dependent benefit: **no**. Method-dependent
benefit: **no**. Adaptation settings with a
reliable matched-FS0 block Macro-F1 improvement: **0**.
The primary gain survives the declared block and seed robustness criteria:
**no**.

The historical head recipe reached its preregistered support-fit validity
threshold in 220 of 525 episodes; head results failing that
check remain descriptive and are not used to establish the main classification.
The result is reported as a target-labelled diagnostic and does not alter the historical comparison baseline.
