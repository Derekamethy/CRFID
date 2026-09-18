# Result definition

Class order is integer labels 0-6 (TagID minus one). Accuracy is the diagonal
sum divided by 3,150. Per-class F1 uses `2TP/(2TP+FP+FN)` with zero when its
denominator is zero; Macro-F1 is the arithmetic mean over all seven classes.
Every seed has equal weight. Reported uncertainty is population standard
deviation over five seeds (`ddof=0`). Predictions are `argmax` over seven
float32 logits in the frozen sample order.
