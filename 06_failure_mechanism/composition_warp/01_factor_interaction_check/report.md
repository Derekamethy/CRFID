# Section 5 — Factor-interaction check

Sub-module: `06_failure_mechanism/composition_warp/01_factor_interaction_check`.
Domains used: **P1, P2, P3 only**. No P4 measurement, label or summary
statistic was read by this sub-experiment.

## Question

The additive composition hypothesis predicts an unobserved factor cell from
the marginal effects of the observed cells. That prediction is only defensible
if the factors it composes do not interact. This sub-experiment measures the
interaction directly, on factor pairs that *are* fully crossed in the source
data, as a proxy for the distance × angle interaction that cannot be measured
without P4.

## Method

For each fully crossed pair, cell means are taken over the 281 ordered signal
positions in the raw dB domain. The two-way decomposition is

```
grand(f)         = mean over all cells
alpha[a](f)      = rowmean[a](f) - grand(f)
beta[b](f)       = colmean[b](f) - grand(f)
additive[a,b](f) = grand(f) + alpha[a](f) + beta[b](f)
gamma[a,b](f)    = cell[a,b](f) - additive[a,b](f)
```

and the summary scalars are the RMS of `gamma` over all cells and points,
against the RMS of each main effect. The design is balanced: every cell of
both pairs holds exactly 1050 samples.

## Pre-registered threshold

`interaction RMS / min(main-effect RMS) > 0.3` ⇒ `HIGH`,
otherwise `LOW`. Fixed in `preregistration.md` before this script was run and
not revised afterwards.

## Result

| Factor pair | RMS interaction (dB) | RMS main A (dB) | RMS main B (dB) | Ratio | Verdict |
|---|---:|---:|---:|---:|:---:|
| `surface_x_position` | 1.901321 | 1.693529 | 4.133041 | **1.1227** | **HIGH** |
| `encoding_state_x_position` | 1.282957 | 0.739388 | 4.133041 | **1.7352** | **HIGH** |

Overall verdict: **HIGH**.

**Reading.** Both ratios exceed 1, not merely the 0.30 threshold. The
interaction term is *larger* than the smaller of the two main effects it is
supposed to be a second-order correction to, so on this data the additive part of
the decomposition does not dominate the residual. `RMS main B` is identical in
both rows because `position` is factor B of both pairs. Note that these two pairs
are proxies: the distance × angle interaction itself cannot be measured without
P4, which is exactly the cell the hypothesis is trying to predict.

Largest single absolute interaction per pair: `surface_x_position` 7.271127 dB, `encoding_state_x_position` 4.541069 dB.

### Per-TagID breakdown (secondary)

The threshold is applied to the pooled value above. The same decomposition
repeated within each TagID gives:

| Factor pair | min ratio | max ratio | verdicts present |
|---|---:|---:|---|
| `surface_x_position` | 0.8720 | 1.2561 | `HIGH` |
| `encoding_state_x_position` | 1.0807 | 1.8450 | `HIGH` |

## Outputs

* `results/interaction_summary.json` — scalars, per-TagID breakdown, threshold, provenance digests.
* `results/interaction_summary.csv` — the raw dB-domain interaction curve: every cell of every
  pair at all 281 ordered signal positions, with its cell mean, additive prediction and residual.
* `plots/interaction_vs_frequency.png` — RMS interaction magnitude against conditional frequency,
  with the pre-registered threshold drawn as a dashed line per pair.

The conditional 5–8 GHz axis is used for the horizontal plot scale only. No
physical frequency axis is asserted; the decomposition itself is computed on
ordered signal positions and does not depend on the axis.

## Scope

Diagnostic and mechanistic only. These numbers are not a Strict-DG, Few-Shot or
External-Data1 result and are not inserted into any existing comparison table.
