# Final results

> Summary of the retained scientific evidence; information-access limitations apply.

| Method | Runs | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majority class 0 | 1 | 0.297297297297 | 0.114583333333 |
| PCA + logistic regression | 1 | 0.996216216216 | 0.994822742489 |
| Raw-signal CNN mean | 5 | 0.748864864865 | 0.678839555377 |
| First-difference CNN mean | 5 | 0.693081081081 | 0.607359527633 |

CNN means are equal-weight seed means. Branch A publishes both sample (`ddof=1`) and
population (`ddof=0`) SD in its frozen aggregate file; any comparison must name the
convention.
