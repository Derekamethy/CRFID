# External Data1 protocol

Set 3 is the sole training set (5,600 rows). Sets 4–9 are the held-out test
sets (1,850 rows). Sets 1–2 are excluded because the dataset documentation
assigns sets 1–3 to different applications and designates the next six sets for
testing classifiers trained with set 3.

The frozen candidates are:

- majority baseline with the four-way tie resolved to class 0;
- training-only StandardScaler, full-SVD PCA retaining at least 95% variance,
  and multinomial logistic regression;
- raw-signal four-class GroupNorm CNN;
- first-difference four-class GroupNorm CNN.

CNN seeds are `[42,43,44,45,46]`. Final epochs are respectively
`[13,13,9,10,9]`. No validation split, early stopping, test selection, or
test-fitted preprocessing is permitted.
