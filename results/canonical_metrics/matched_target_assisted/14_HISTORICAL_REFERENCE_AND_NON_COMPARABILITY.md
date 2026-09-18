
# Historical reference and non-comparability

`HISTORICAL_RETROSPECTIVE_P4_REFERENCE_NOT_HELD_OUT_NOT_DIRECTLY_COMPARABLE`

The historical retrospective P4 result of 0.5771
Accuracy and 0.5836 Macro-F1 is retained as a
non-held-out, P4-informed historical reference. It is not used to estimate the
causal effect of target assistance and is not directly compared as a matched
treatment effect.

Provenance: `Canonical V2 retrospective P4-informed reference; full-P4 outcomes influenced method promotion and the reported partition was not an independent held-out P4 test.`.

The new A0 is the fair source-only control because it shares the representation,
checkpoint, split, and final-test sample order with the treatment arms. A1 is
the fair analogue of target-informed method promotion because only P4-Val
selects among already frozen source-fitted candidates. A2 measures limited
labelled adaptation with the encoder still frozen. The historical reference
uses a methodologically separate evaluation and development lineage.

## Explicit non-comparability statement

- The historical retrospective figures (Accuracy 0.5771428571, Macro-F1
  0.5835693346) remain non-held-out and P4-informed; they are not an
  independent test of any model reported here.
- They are not paired with the canonical Strict-DG source-only result
  (Accuracy 0.1566, Macro-F1 0.1122) at the unit level, and this branch does
  not use them to estimate the causal effect of target assistance.
- This matched branch — A0, A1, and A2 sharing one encoder, checkpoint,
  candidate set, split, and test samples — does **not** make the historical
  0.1566 and 0.5771 directly comparable. It provides a fair comparison **among
  A0, A1, and A2 only**. Any apparent "gap closed" between 0.1566 and 0.5771
  cannot be attributed to this branch's readout-selection or adaptation
  mechanisms: giving this matched pipeline P4-Val-informed selection over
  three source-fitted readouts moved Macro-F1 by only +0.0023 (95% CI −0.0011
  to +0.0067), far short of that historical gap, and the historical pipeline
  itself was not re-derived or decomposed by this branch. Treat this as
  suggestive context, not a causal attribution.
