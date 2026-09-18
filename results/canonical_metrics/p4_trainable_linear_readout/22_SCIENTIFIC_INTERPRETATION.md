# Scientific interpretation

Classification: `REPRESENTATION_BOTTLENECK_SUPPORTED`.

The largest point difference between the trainable linear readout and the matched
cosine prototype occurred at 500 labels:
+0.0404 Macro-F1, 95% CI
[-0.0409,
0.1165]. At 500
labels the linear readout achieved Macro-F1 0.1427,
compared with prototype 0.1023. The response was
not monotonic across the registered budgets.

At 500 labels, mean support Macro-F1 was 0.2281
while held-condition query Macro-F1 was 0.1427,
a support-minus-query gap of 0.0854.
This contrast distinguishes fitting observed P4 conditions from transfer to unseen
ER-by-surface blocks; it was not used for fitting or model selection.

TagID 1 had linear F1 0.0815 at 500 labels and changed
+0.0813 relative to the prototype. TagID 6 had
linear F1 0.1254 and changed
-0.1481. The weakest final class was TagID
4 (F1 0.0435). The weakest
reported physical stratum was ER=0
(linear Macro-F1 0.0869).

The evidence therefore attributes the earlier negative curve according to the sealed
classification rule above. It does not prove a causal representation defect, and it
does not authorize encoder fine-tuning within this branch.
