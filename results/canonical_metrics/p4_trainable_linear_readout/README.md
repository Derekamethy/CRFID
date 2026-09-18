# P4 Trainable Linear-Readout Calibration Curve v1

## A. FINAL CLASSIFICATION

`REPRESENTATION_BOTTLENECK_SUPPORTED`

## B. SCIENTIFIC IDENTITY

- Preregistration SHA-256: `3981144793981f96b44c486d413f32436129427902bbad34b6974257de57836b`
- Parent result commit: `5af4b691d92a8949d78883d8e468cd008cc1aae1`
- Frozen C1 source seeds: 42--46
- Inherited fold manifest: `81b86e65bce6ff35e04b0b6a3fe2310fa5a65173e9d86c6693d40ec9e22c0482`
- Inherited support manifest: `47eacce86bbe1426e780beb9949125130c8348430d4711039f9c6099aaf6eae3`

## C. VALIDITY

All 14 pre-execution gates passed. All encoder states remained frozen, every
support prefix was row/block/exact-signal disjoint from its query, the regularisation
value was selected from P1--P3 only, and 2700
positive-budget predictions were frozen before query-label scoring. The deterministic
refit probe passed with head hash `d75db73d91cdeb1fc8709c2110a2f4d3cf2d05003e994c9cd6e4a25f1a3f282b`.

## D. PRIMARY RESULTS

| Budget | Blocks | Prototype Macro-F1 | Linear Macro-F1 | Delta linear-prototype [95% CI] | Linear Accuracy | Delta linear-zero |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.1122 | 0.1122 | +0.0000 [0.0000, 0.0000] | 0.1566 | +0.0000 |
| 7 | 7.0 | 0.1271 | 0.1267 | -0.0004 [-0.0454, 0.0459] | 0.1341 | +0.0145 |
| 10 | 10.0 | 0.1306 | 0.1261 | -0.0045 [-0.0463, 0.0396] | 0.1381 | +0.0139 |
| 20 | 20.0 | 0.1356 | 0.1364 | +0.0008 [-0.0584, 0.0637] | 0.1402 | +0.0241 |
| 21 | 21.0 | 0.1337 | 0.1374 | +0.0037 [-0.0585, 0.0697] | 0.1399 | +0.0252 |
| 35 | 35.0 | 0.1196 | 0.1447 | +0.0250 [-0.0482, 0.0993] | 0.1463 | +0.0324 |
| 50 | 42.0 | 0.1147 | 0.1405 | +0.0258 [-0.0472, 0.1005] | 0.1426 | +0.0282 |
| 100 | 42.0 | 0.1099 | 0.1443 | +0.0343 [-0.0428, 0.1110] | 0.1455 | +0.0320 |
| 200 | 42.0 | 0.1078 | 0.1413 | +0.0335 [-0.0472, 0.1089] | 0.1423 | +0.0291 |
| 500 | 42.0 | 0.1023 | 0.1427 | +0.0404 [-0.0409, 0.1165] | 0.1434 | +0.0305 |

## E. LINEAR VS PROTOTYPE

The largest paired point difference was +0.0404
at 500 labels, with 95% CI
[-0.0409,
0.1165].

## F. LABEL-BUDGET RESPONSE

The linear curve was not monotonic.
Its crossed-unit Macro-F1 SD changed from 0.0336
at 7 labels to 0.0248 at 500 labels.

## G. TRAIN VS HELD-CONDITION GAP

At 500 labels: support Macro-F1 0.2281,
held-condition Macro-F1 0.1427, gap
0.0854.

## H. CLASS / CONDITION DIAGNOSTICS

At 500 labels TagID 1 had F1 0.0815; TagID 6 had F1
0.1254. The weakest class was TagID 4.
The weakest reported stratum was ER=0.

## I. SCIENTIFIC INTERPRETATION

See `22_SCIENTIFIC_INTERPRETATION.md`. Positive budgets are target-assisted and do
not constitute strict source-only domain generalisation.

## J. RELATION TO PREVIOUS CALIBRATION CURVE

The parent folds, supports, encoders, queries, seeds, budgets, metrics, and bootstrap
draws are unchanged. The sole scientific delta is a trainable linear softmax readout
in place of the target-only cosine prototype.

## K. THESIS-READY CONCLUSION

Following the negative frozen-C1 cosine-prototype calibration curve, a matched
block-disjoint control replaced only the target adapter with a source-anchored
trainable linear classifier. The resulting evidence was classified as
`REPRESENTATION_BOTTLENECK_SUPPORTED`. At 500 labels the linear readout achieved Macro-F1
0.1427, versus 0.1023
for the matched prototype, while its support-to-held-condition gap was
0.0854. This diagnostic constrains
whether the previous failure is attributable to prototype geometry, frozen
representation geometry, or both, without changing the encoder or using query labels
for adaptation.

## L. NEXT-EXPERIMENT DECISION

`ENCODER_FINETUNING_DIAGNOSTIC_JUSTIFIED`

Encoder fine-tuning was not executed.
