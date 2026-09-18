# Source-only fine-tuning hyperparameter selection

Selection used only the audited P1--P3 position-held pseudo-target protocol. No P4 file, support, prediction, or metric was opened.

| Arm | Learning rate | Encoder weight decay | Epochs | Worst-position Macro-F1 | Overall Macro-F1 |
|---|---:|---:|---:|---:|---:|
| PARTIAL_FT | 0.0003 | 0 | 10 | 0.219348 | 0.224833 |
| FULL_FT | 0.0001 | 0.0001 | 20 | 0.214145 | 0.222157 |
