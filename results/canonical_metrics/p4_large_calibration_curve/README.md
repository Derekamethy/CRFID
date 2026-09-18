# Larger P4 Labelled-Target Calibration Curve

## A. FINAL CLASSIFICATION

`WEAK_OR_UNRELIABLE_TARGET_CALIBRATION_GAIN`

## B. EXPERIMENT IDENTITY

- Preregistration SHA-256: `30e65854c09773f3ff64054cf04fabc215a9e9fa27919ffc9de448d1f48bb99d`
- Data lineage: governed A1/A2/A3 P4 hashes in `01_EXACT_CONFIGURATION.json`
- Model lineage: five frozen source-only C1 checkpoints, seeds 42-46

## C. CALIBRATION DESIGN

- Total-label budgets: 0, 7, 10, 20, 21, 35, 50, 100, 200, 500
- Canonical matched anchors: 7 = 1-shot, 21 = 3-shot, 35 = 5-shot
- Independent unit: `TagID x ER x surface x P4` physical condition block
- Support/query: nested support prefixes; three fixed block-disjoint Latin-square query folds; all 3,150 P4 rows queried out of fold
- Seeds: five source x twenty support
- Uncertainty: 10,000-replicate paired TagID-stratified block x crossed source/support-seed bootstrap

Every positive budget is target-assisted. Only zero labels is strict source-only DG.

## D. PRIMARY RESULTS

| P4 labelled budget | Independent blocks | Macro-F1 | 95% CI | Delta vs 0 | Accuracy | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 labels (strict source-only DG) | 0.0 | 0.1122 | [0.0612, 0.1620] | +0.0000 | 0.1566 | [0.0997, 0.2150] |
| 1-shot (7 labels) | 7.0 | 0.1271 | [0.0998, 0.1521] | +0.0149 | 0.1407 | [0.1129, 0.1713] |
| 10 labels | 10.0 | 0.1306 | [0.1022, 0.1549] | +0.0184 | 0.1425 | [0.1155, 0.1720] |
| 20 labels | 20.0 | 0.1356 | [0.0995, 0.1674] | +0.0233 | 0.1446 | [0.1082, 0.1823] |
| 3-shot (21 labels) | 21.0 | 0.1337 | [0.0979, 0.1661] | +0.0214 | 0.1432 | [0.1074, 0.1819] |
| 5-shot (35 labels) | 35.0 | 0.1196 | [0.0801, 0.1586] | +0.0074 | 0.1310 | [0.0901, 0.1750] |
| 50 labels | 42.0 | 0.1147 | [0.0728, 0.1568] | +0.0025 | 0.1265 | [0.0835, 0.1733] |
| 100 labels | 42.0 | 0.1099 | [0.0670, 0.1527] | -0.0023 | 0.1235 | [0.0794, 0.1709] |
| 200 labels | 42.0 | 0.1078 | [0.0654, 0.1509] | -0.0044 | 0.1223 | [0.0769, 0.1714] |
| 500 labels | 42.0 | 0.1023 | [0.0603, 0.1465] | -0.0099 | 0.1176 | [0.0726, 0.1659] |

## E. KEY THRESHOLD

`NONE`. Practical-only first budget: `None`. Reliable-only first budget: `None`.

## F. STRICT-DG -> TARGET-ASSISTED GAP

Recovered fractions relative to the historical 0.5836 retrospective result or the
0.9835 matched within-condition result are `NOT_IDENTIFIABLE_UNDER_MATCHED_PROTOCOL`.
Those references used non-comparable selection, splitting, and dependence structures.
The directly valid quantities are the paired budget-minus-zero changes in the table.

## G. CONDITION-COVERAGE FINDING

No observed budget met the preregistered combined threshold of a +0.05 Macro-F1 point gain with a paired interval excluding zero. The preregistered coverage-dominance rule was not met; condition coverage and raw count cannot be cleanly separated below the 42-block ceiling, and the post-coverage contrast is reported descriptively. TagID 6 had the largest descriptive 0-to-500 class-F1 gain (+0.2710); TagID 1 remained weakest at 500 labels (F1 0.0002).

- 7-to-35 Macro-F1 change: -0.0075, 95% CI [-0.0429, 0.0294]
- 50-to-500 Macro-F1 change at fixed 42-block coverage: -0.0123, 95% CI [-0.0317, 0.0057]
- At 500 labels, weakest ER level: 1 (Macro-F1 0.0594); weakest surface: 2 (Macro-F1 0.0590).

## H. LEAKAGE / VALIDITY AUDIT

All 16 critical gates passed. Supports and queries are row-, physical-block-,
and exact-signal-disjoint within every fold; all 50 repeated acquisitions remain in one
role. Every prediction was frozen, SHA-256 registered, and persisted before the global
query-label barrier opened. Frozen encoder states were unchanged. P1-P3 preprocessing
was not refit. Query labels influenced final scoring only.

## I. RELATION TO PRIOR P4 EXPERIMENTS

- Strict-DG: exactly reproduced as the zero-label anchor.
- Historical few-shot: preserved as external 1/3/5-shot provenance; not copied because its episode/query and Euclidean protocol differ.
- Factor-aware few-shot: supplies the matched block-disjoint folds, frozen encoders, and cosine-prototype lineage.
- Retrospective P4: external, full-P4 outcome-informed reference only.
- Matched target-assisted P4: external within-condition, near-duplicate-inflated reference only.

## J. THESIS-READY CONCLUSION

Using a frozen P1-P3-trained C1 encoder and a fixed cosine-prototype adapter, the block-disjoint P4 labelled-target calibration curve was classified as `WEAK_OR_UNRELIABLE_TARGET_CALIBRATION_GAIN`. The strict source-only anchor had Macro-F1 0.1122; at 500 target labels the target-assisted Macro-F1 was 0.1023 (paired change -0.0099, 95% CI [-0.0734, 0.0552]). No observed budget met the preregistered combined threshold of a +0.05 Macro-F1 point gain with a paired interval excluding zero. Positive-budget results quantify labelled-target calibration to held P4 condition blocks and are not source-only domain generalisation.

## K. REMAINING ISSUES

- There is no protocol-comparable fully target-assisted endpoint, so gap-recovery percentages are unavailable.
- The 63 blocks come from one P4 corpus rather than an independent acquisition campaign.
- Below 42 support blocks, raw count and condition coverage increase together; their effects are not causally separable.
- Findings are conditional on the frozen C1 representation and cosine-prototype adapter.

## Scientific question

Yes. The requested larger curve has been estimated at exact total-label budgets 10,
20, 50, 100, 200, and 500 in addition to matched 1/3/5-shot anchors. Budget means raw
labelled measurements; the accompanying coverage columns prevent those counts from
being mistaken for independent conditions. The first reliably and practically
beneficial budget is `NONE`. See `16_INCREMENTAL_GAINS.csv`,
`17_SUPPORT_COVERAGE_SUMMARY.csv`, and `22_LIMITATIONS.md` for diminishing-return,
coverage, deployment, and limitation details.
