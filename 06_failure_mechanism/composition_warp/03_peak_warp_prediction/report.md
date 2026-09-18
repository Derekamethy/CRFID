# Section 7 — Peak-warp prediction and predicted collisions

Sub-module: `06_failure_mechanism/composition_warp/03_peak_warp_prediction`.
**No raw P4 data was read.** The only P4-derived information used is the
conclusion text of the existing class-collapse finding — which TagIDs collapse —
which is the single comparison-only read permitted for this sub-experiment.

## Question

If the additive composition hypothesis holds for resonance geometry, the P4
peak positions are predictable from the source domains alone. Tag identity in
this system is carried by relative resonance geometry, so tags whose predicted
P4 peaks land on top of each other are the tags a classifier should be unable
to tell apart. That gives a source-only prediction of *which classes collapse*,
testable against a conclusion that was reached from P4.

## Method

Peak extraction is reused by import from
`crfid.analysis.reference_peaks.strongest_window_peaks` in `mode = "min"`,
applied to the mean spectrum of each `(tag_id, er, surface, position)` block of
50 repetitions, on the four verified resonator windows:

| Window | Conditional range (GHz) |
|---|---|
| `Lc` | 5.100 – 5.745 |
| `L3` | 5.800 – 6.350 |
| `L2` | 6.450 – 6.950 |
| `L1` | 7.050 – 7.700 |

756 extractions were performed; 0.8585 of them landed on an
interior local minimum rather than a window edge.

> **Window naming.** The task specification labels these `Lc≈5.30, L1≈5.84,
> L2≈6.46, L3≈7.17`. The repository's own reference-peak report assigns the two
> outer regions the other way round — `L3` is the 5.800–6.350 band and `L1` is
> the 7.050–7.700 band. The four *regions* are the same; only the `L1`/`L3`
> labels differ. The repository's naming is used here, because §7 is required to
> reuse the reference-peak branch rather than re-derive it.

Then, per TagID and per `(er, surface)` condition:

```
delta_f_distance = peak(P3) - peak(P1)
delta_f_angle    = peak(P2) - peak(P1)
peak_hat(P4) = peak(P1) + delta_f_distance + delta_f_angle
```

Per-TagID predicted positions are the mean over the nine (er, surface) conditions of that tag.

## Pre-registered collision threshold

Delta_min is the smallest per-tag mean peak separation, over all source positions and peaks, among tag pairs whose separation exceeds their pooled within-tag standard deviation across the nine (er, surface) conditions.

`Delta_min = 0.170238 GHz`, attained at position `P1`, peak `L2`, tag pair [2, 6]. 13 of 252 pair/peak/position combinations were reliably separated in-domain (0.0516).

**Collision threshold = 1.5 × Delta_min = 0.255357 GHz.** Fixed in `preregistration.md` before this
script was run and not revised afterwards.

## Predicted P4 peak positions (GHz, conditional axis)

| TagID | `Lc` | `L3` | `L2` | `L1` |
|---:|---:|---:|---:|---:|
| 1 | 5.2821 | 6.1893 | 6.5940 | 7.2607 |
| 2 | 5.4762 | 6.1655 | 6.5869 | 7.3155 |
| 3 | 5.2810 | 6.2988 | 6.5179 | 7.1798 |
| 4 | 5.3595 | 6.0833 | 6.5250 | 7.4488 |
| 5 | 5.3345 | 6.1512 | 6.5905 | 7.2726 |
| 6 | 5.3560 | 6.1369 | 6.7262 | 7.4119 |
| 7 | 5.4655 | 6.1226 | 6.5405 | 7.4286 |

## Predicted collisions

**21 of 21 tag pairs are flagged as predicted collisions**: (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (2,3), (2,4), (2,5), (2,6), (2,7), (3,4), (3,5), (3,6), (3,7), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7).

| Pair | min predicted separation (GHz) | collided peaks | predicted collision |
|---|---:|---|:---:|
| (1,2) | 0.007143 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (1,3) | 0.001190 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (1,4) | 0.069048 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (1,5) | 0.003571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (1,6) | 0.052381 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (1,7) | 0.053571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (2,3) | 0.069048 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (2,4) | 0.061905 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (2,5) | 0.003571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (2,6) | 0.028571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (2,7) | 0.010714 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (3,4) | 0.007143 | `Lc`, `L3`, `L2` | **yes** |
| (3,5) | 0.053571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (3,6) | 0.075000 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (3,7) | 0.022619 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (4,5) | 0.025000 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (4,6) | 0.003571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (4,7) | 0.015476 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (5,6) | 0.014286 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (5,7) | 0.028571 | `Lc`, `L3`, `L2`, `L1` | **yes** |
| (6,7) | 0.014286 | `Lc`, `L3`, `L2`, `L1` | **yes** |

### Discriminative power of this test

**The test flags all 21 of 21 pairs, so it is not discriminative.** Any
hit rate it achieves against any observed collapse set is attained trivially,
with a false-alarm rate of 100 % on the complement. The hit counts below are
reported because the specification asks for them, but they are **not evidence**
that the additive peak-warp prediction identifies the collapsing tags.

The reason is visible in the threshold derivation, and it is the substantive
finding of this sub-experiment: only 13 of 252 pair/peak/position combinations
(5.16%) are reliably separated in-domain at all. Across
the nine `(er, surface)` conditions of a tag, the within-tag scatter of a peak
position is larger than the between-tag separation almost everywhere. The
resulting `Delta_min` of 0.1702 GHz is therefore large — about 16 axis steps —
while the predicted P4 separations are an order of magnitude smaller.

Read plainly: **the four reference peaks do not separate the seven TagIDs in the
source domains in the first place**, so composing their shifts cannot single out
TagIDs 5/6/7. That is a statement about the reference-peak representation, not a
refutation of the collapse conclusion.

## Comparison against the existing class-collapse conclusion

Comparison-only read of existing P4-derived *conclusions*. No raw P4 data,
spectra or embeddings were read.

### `few_shot_tracked_collapse_triple` (primary)

TagIDs: [5, 6, 7]. Source: `historical few-shot P4 collapse summary retained in this diagnostic`.

> The few-shot branch tracks Tags 5, 6 and 7 as the collapse set under P4 adaptation. This is the set named by the task specification.

Hits **3/3** of the observed collapse pairs; 18 false alarms, 0 misses, 0 correct rejections out of 21 pairs.

| Pair | predicted collision | in observed collapse set | outcome |
|---|:---:|:---:|---|
| (1,2) | yes | no | `false_alarm` |
| (1,3) | yes | no | `false_alarm` |
| (1,4) | yes | no | `false_alarm` |
| (1,5) | yes | no | `false_alarm` |
| (1,6) | yes | no | `false_alarm` |
| (1,7) | yes | no | `false_alarm` |
| (2,3) | yes | no | `false_alarm` |
| (2,4) | yes | no | `false_alarm` |
| (2,5) | yes | no | `false_alarm` |
| (2,6) | yes | no | `false_alarm` |
| (2,7) | yes | no | `false_alarm` |
| (3,4) | yes | no | `false_alarm` |
| (3,5) | yes | no | `false_alarm` |
| (3,6) | yes | no | `false_alarm` |
| (3,7) | yes | no | `false_alarm` |
| (4,5) | yes | no | `false_alarm` |
| (4,6) | yes | no | `false_alarm` |
| (4,7) | yes | no | `false_alarm` |
| (5,6) | yes | yes | `hit` |
| (5,7) | yes | yes | `hit` |
| (6,7) | yes | yes | `hit` |

Pairs that are neither predicted nor observed are omitted from the table and
counted as correct rejections (0).

### `strict_dg_zero_recall_set` (secondary)

TagIDs: [3, 5, 6, 7]. Source: `historical Strict-DG zero-recall summary retained in this diagnostic`.

> Tags 3, 5, 6, and 7 exhibit at least one zero-recall seed; Tag 6 is the most severe pooled failure (recall 0.00178, 4 correct of 2,250).

Hits **6/6** of the observed collapse pairs; 15 false alarms, 0 misses, 0 correct rejections out of 21 pairs.

| Pair | predicted collision | in observed collapse set | outcome |
|---|:---:|:---:|---|
| (1,2) | yes | no | `false_alarm` |
| (1,3) | yes | no | `false_alarm` |
| (1,4) | yes | no | `false_alarm` |
| (1,5) | yes | no | `false_alarm` |
| (1,6) | yes | no | `false_alarm` |
| (1,7) | yes | no | `false_alarm` |
| (2,3) | yes | no | `false_alarm` |
| (2,4) | yes | no | `false_alarm` |
| (2,5) | yes | no | `false_alarm` |
| (2,6) | yes | no | `false_alarm` |
| (2,7) | yes | no | `false_alarm` |
| (3,4) | yes | no | `false_alarm` |
| (3,5) | yes | yes | `hit` |
| (3,6) | yes | yes | `hit` |
| (3,7) | yes | yes | `hit` |
| (4,5) | yes | no | `false_alarm` |
| (4,6) | yes | no | `false_alarm` |
| (4,7) | yes | no | `false_alarm` |
| (5,6) | yes | yes | `hit` |
| (5,7) | yes | yes | `hit` |
| (6,7) | yes | yes | `hit` |

Pairs that are neither predicted nor observed are omitted from the table and
counted as correct rejections (0).

## Outputs

* `results/predicted_p4_peaks.json` — extraction settings, observed per-TagID peak means and
  within-tag spreads per source position, per-condition distance/angle shifts, and the predicted
  P4 positions.
* `results/collision_prediction_vs_observed.json` — the pre-registered threshold with its
  derivation, all 21 pairwise predictions, and the hit/miss comparison against both recorded
  class-collapse conclusions.
* `plots/predicted_vs_actual_peaks.png` — observed P1/P2/P3 peaks and the predicted P4 peak per
  TagID, with the collision width drawn around each prediction.

## Scope

Every frequency in this report is on the **conditional** 281-point 5–8 GHz axis.
The reference-peak branch release status is `BLOCKED` and the Strict-DG
configuration records `PHYSICAL_FREQUENCY_AXIS_PROVENANCE_UNRESOLVED`. No
physical frequency axis is asserted. Diagnostic and mechanistic only; not
inserted into any existing comparison table.
