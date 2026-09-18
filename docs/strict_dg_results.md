# Strict-DG results

The source-only selector chose `C1_FIRST_DIFFERENCE_ERM_1DCNN`. P4 was held outside preprocessing, hyperparameter selection, architecture selection, epoch selection, and checkpoint selection.

| Seed | Epochs | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 42 | 13 | 0.172698 | 0.123054 |
| 43 | 13 | 0.119365 | 0.096557 |
| 44 | 9 | 0.129206 | 0.100795 |
| 45 | 10 | 0.204127 | 0.144565 |
| 46 | 9 | 0.157778 | 0.096184 |

Equal-weight mean Accuracy is **0.156635 ± 0.030516** and mean Macro-F1 is **0.112231 ± 0.018956** across the five fixed seeds.

## Verification

The retained checkpoints, preprocessing state, confusion matrices, and final metric tables are mutually consistent. Recomputed predictions and metrics match the frozen result identities recorded for the study.

The public repository keeps the five selected checkpoints and compact result evidence, but not the raw Tyndall/Paper4 measurements or the large historical prediction/logit replay bundles.

## Interpretation

Seven-class balanced chance accuracy is approximately **0.1429**. The selected source-only model therefore transfers poorly to the held P4 geometry, with several classes collapsing to zero recall.

This is a bounded result for the tested source-selected representation, model family, data campaign, and P4 geometry. It does not establish that CRFID domain generalization is impossible, and it is not directly comparable with retrospective P4-informed results that use target outcomes during selection.
