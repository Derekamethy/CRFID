# Domain-aware mixup executive summary

Governed runtime binding succeeded. Synthetic implementation validity passed, including gradient flow through both parents, encoder, and head. The governed pairing gate failed: all three canonical inner-training folds had 2,100/4,200 eligible parents (50.0%), below the required >99%.

The raw dataset contains complete TagID × ER × surface factorial support at every position: 7 TagIDs, 3 ER levels, 3 surfaces, 63 condition blocks, and 3,150 rows per position. The 50% pairing limitation arises only inside the frozen canonical `inner_train` subsets. In each LOPO fold, each of the two source positions retains 42 condition keys for training, but only 21 are shared across both source-position subsets; the matched blocks for the remaining keys exist in the raw dataset at the other source position but were assigned to `inner_validation`. Thus, 21 × 2 × 50 = 2,100 eligible parent samples out of 42 × 2 × 50 = 4,200 are available for strictly matched cross-position mixup. The root cause is `SPLIT_INDUCED_CROSS_POSITION_PAIRING_INCOMPATIBILITY`, not incomplete raw-data support.

Development runs completed: `0/75`. Final runs completed: `0/15`. Selected alpha: `not available`. P4 artifacts and labels were not opened. Source, probe, geometry, P4, uncertainty, diagnostic-classification, and main-classification results are not available and have not been fabricated.

Stop status: `FAIL_DOMAIN_AWARE_PAIRING_COVERAGE`.

Focused implementation checks passed. The scientific branch stopped at the preregistered pairing-coverage gate before development training or P4 evaluation.
