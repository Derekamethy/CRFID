# External Data1 portability

This supporting study tests whether the processing and modelling pipeline can operate on an independent four-class external corpus.

Headline within-Data1 results:
- PCA + logistic regression: Accuracy **0.9962**, Macro-F1 **0.9948**
- Raw-signal CNN mean: Accuracy **0.7489**, Macro-F1 **0.6788**
- First-difference CNN mean: Accuracy **0.6931**, Macro-F1 **0.6074**

These are **within-Data1** results. They do not establish seven-class Tyndall/Paper4 validation, cross-dataset label equivalence, or deployment transfer.