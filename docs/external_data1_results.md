# External Data1 results

| Method | Accuracy mean | Macro-F1 mean | Macro-F1 sample SD |
|---|---:|---:|---:|
| Majority baseline | 0.2972972972972973 | 0.11458333333333333 | 0 |
| PCA + logistic regression | 0.9962162162162163 | 0.9948227424886364 | 0 |
| Raw CNN, five seeds | 0.7488648648648649 | 0.6788395553773673 | 0.009766316246323152 |
| First-difference CNN, five seeds | 0.693081081081081 | 0.6073595276333663 | 0.017369844763141087 |

All comparisons are against a frozen set-3 versus sets-4–9 protocol. The PCA
result supports clear learnability on this split; it is not evidence of
deployment or cross-dataset generalization. The raw CNN supports pipeline
portability. First difference is not supported as superior on Data1.
