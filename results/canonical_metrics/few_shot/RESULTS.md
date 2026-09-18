# Few-Shot P4 Results

All values are means over 18 episodes of five-seed episode means, with the
population standard deviation over the 18 episode means in parentheses. Every
row is evaluated on the same 1,400-sample query population.

## Separate canonical curves

| Method | Shot | Support per episode | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| `FROZEN_SOURCE_HEAD` (zero-shot) | 0 | 0 | 0.156635 (0.023472) | 0.107257 (0.025602) |
| `TARGET_ONLY_SQUARED_EUCLIDEAN_PROTOTYPES` | 1 | 7 | 0.142698 (0.037640) | 0.112159 (0.033743) |
| `TARGET_ONLY_SQUARED_EUCLIDEAN_PROTOTYPES` | 3 | 21 | 0.165429 (0.021727) | 0.145141 (0.023730) |
| `TARGET_ONLY_SQUARED_EUCLIDEAN_PROTOTYPES` | 5 | 35 | 0.162635 (0.029901) | 0.144360 (0.029465) |
| `FROZEN_ENCODER_SOURCE_ANCHORED_LINEAR_HEAD_ADAPTATION` | 1 | 7 | 0.147476 (0.031605) | 0.120852 (0.027799) |
| `FROZEN_ENCODER_SOURCE_ANCHORED_LINEAR_HEAD_ADAPTATION` | 3 | 21 | 0.149754 (0.024019) | 0.129295 (0.021783) |
| `FROZEN_ENCODER_SOURCE_ANCHORED_LINEAR_HEAD_ADAPTATION` | 5 | 35 | 0.157651 (0.040010) | 0.138984 (0.036008) |

The zero-shot row is a single shared reference reused by both curves, not two
independent results.

## Paired episode deltas

Deterministic percentile bootstrap over 18 paired episodes, 10,000 resamples,
seed 20260720. These are supporting descriptive intervals.

| Comparison | Mean macro-F1 delta | 95% interval | Positive / tie / negative |
|---|---:|---:|---:|
| Prototype 0 to 1 | 0.004902 | [-0.012224, 0.020574] | 11 / 0 / 7 |
| Prototype 0 to 3 | 0.037884 | [0.022224, 0.056097] | 17 / 0 / 1 |
| Prototype 0 to 5 | 0.037103 | [0.019806, 0.054516] | 16 / 0 / 2 |
| Head 0 to 1 | 0.013594 | [0.000978, 0.025817] | 12 / 0 / 6 |
| Head 0 to 3 | 0.022038 | [0.011519, 0.032641] | 16 / 0 / 2 |
| Head 0 to 5 | 0.031727 | [0.012092, 0.050825] | 14 / 0 / 4 |
| Head minus prototype at 1 | 0.008692 | [-0.005581, 0.022816] | 11 / 0 / 7 |
| Head minus prototype at 3 | -0.015846 | [-0.031515, -0.000718] | 6 / 0 / 12 |
| Head minus prototype at 5 | -0.005376 | [-0.019259, 0.007973] | 8 / 0 / 10 |

## Calibration cost

| Shot | Labelled support samples | Query samples | Support / query |
|---:|---:|---:|---:|
| 1 | 7 | 1400 | 0.5% |
| 3 | 21 | 1400 | 1.5% |
| 5 | 35 | 1400 | 2.5% |

## Per-class behaviour

Recalls are aggregated from persisted unit-level per-class records using the same
five-seed-then-episode rule. The patterns below are descriptive, not causal.

| Class | Zero-shot | Prototype 1 / 3 / 5 | Head 1 / 3 / 5 |
|---|---:|---|---|
| Tag-5 | 0.172889 | 0.093000 / 0.099056 / 0.099944 | 0.131111 / 0.120611 / 0.090722 |
| Tag-6 | 0.001778 | 0.184389 / 0.202611 / 0.259944 | 0.129556 / 0.141222 / 0.140667 |
| Tag-7 | 0.109778 | 0.150389 / 0.199889 / 0.198333 | 0.202111 / 0.239556 / 0.304000 |

Tag-5 is not resolved by either adapter and degrades under the head as shots
increase. Tag-6 rises from a near-zero zero-shot recall under both adapters, more
strongly under prototypes. Tag-7 improves monotonically under the head and
exceeds the prototype recall most clearly at 5-shot.

## Source-side selector transfer

Stage FS4 predicted head superiority at every shot from P1-P3 evidence
(+0.019302, +0.012450, +0.013883 macro-F1). On P4 the head won only at 1-shot
(+0.008692); prototypes won at 3-shot (-0.015846) and 5-shot (-0.005376). Source
pseudo-target validation therefore did not predict the target ranking, which is
evidence that P4 represents a materially different shift from held P1-P3
positions.

## Overfitting evidence

| Shot | Lambda | Weight displacement | Bias displacement | Support CE before | Support CE after | Fraction of units where support improved and query macro-F1 fell |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 2.921789 | 0.063593 | 2.709575 | 0.922688 | 0.455556 |
| 3 | 100 | 2.310673 | 0.048638 | 2.708939 | 1.602053 | 0.344444 |
| 5 | 0.1 | 33.390630 | 0.688550 | 2.716972 | 0.024599 | 0.333333 |

At 5-shot the mean support cross-entropy is near zero while weight displacement
is large, which supports a support-set overfitting risk without establishing
causality.
