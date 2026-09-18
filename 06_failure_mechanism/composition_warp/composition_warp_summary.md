# Composition-warp diagnostic

**This sub-module does not modify or challenge the frozen Strict-DG, Few-Shot,
or External-Data1 results.** Everything below is diagnostic and mechanistic. None
of these numbers belongs in any existing comparison table.

---

## The hypothesis

P1–P4 form a 2 × 2 distance × angle design: P1 = 50 mm/0°, P2 = 50 mm/45°,
P3 = 150 mm/0°, P4 = 150 mm/45°. Both of P4's factor levels are independently
observed in the source domains. The additive / no-interaction hypothesis is that
the unobserved cell is the sum of the observed marginal effects:

```
mu_hat[c, P4](f) = mu[c, P2](f) + mu[c, P3](f) - mu[c, P1](f)
```

per TagID and condition, on paired P1/P2/P3 samples. If it held, P4 behaviour
would be predictable from source data alone. The three sub-experiments test the
hypothesis directly (§5), and carry it into the noise floor (§6) and the
resonance geometry (§7).

Only the *factor-level definitions* of P4 were used. P4 measurements were read
exactly once, in §6, through the existing machine-enforced target gate
(3 file reads, 0 target training steps, 0 target preprocessing refits).
§7 additionally read the *conclusion text* of the existing class-collapse finding.
No other target access occurred.

---

## §5 — Factor interaction: **HIGH**

| Factor pair | RMS interaction (dB) | Smaller main effect (dB) | Ratio | Verdict |
|---|---:|---:|---:|:---:|
| `surface_x_position` | 1.9013 | 1.6935 | **1.1227** | **HIGH** |
| `encoding_state_x_position` | 1.2830 | 0.7394 | **1.7352** | **HIGH** |

Pre-registered threshold: ratio > 0.3 ⇒ `HIGH`. Both fully crossed factor
pairs exceed it, and by a wide margin: in both cases the interaction RMS is
*larger* than the smaller of the two main effects it is supposed to be a
second-order correction to. The same decomposition repeated within each of the
seven TagIDs returns `HIGH` for all seven, in both pairs — ranges 0.872–1.256 and 1.081–1.845.

**The stopping rule therefore fired.** §6 and §7 were still run in full, as
pre-registered, and their interpretive weight is downgraded accordingly. The
§5 threshold was not adjusted after the result was seen.

---

## §6 — Composite noise floor and noise-augmented retraining

Formula used: **`additive_on_variance`** — `sigma2_P2 + sigma2_P3 - sigma2_P1, negatives clipped to zero`.

> ### Documented deviation from the pre-registration
>
> The pre-registered automatic switch fired — additive-on-variance goes negative
> in 0.2745 of cells, above the pre-declared 0.01 limit — and selected the
> `log_variance_additive` fallback. That estimator was computed and then **rejected**
> on a recorded numerical-validity criterion:
> *max estimated noise variance must not exceed the total variance of the measured source signal, and the estimate must be finite everywhere*.
> Its maximum estimate is 1.914e+07 dB², about 4.1e+05× the total variance of the
> measured source signal (46.6932 dB²), because `sigma2_P1` reaches values far
> below the typical repetition variance and the ratio estimator diverges. A noise
> floor larger than the entire dynamic range of the data is not a noise floor.
>
> The primary additive rule was used instead, with its negative cells clipped to
> zero. **This criterion was formulated after seeing the Stage-1 diagnostics and is
> therefore post-hoc.** It is recorded here and in
> `02_noise_floor_composition/results/noise_floor_estimates.json`, which reports
> both estimators in full, rather than applied silently. The pre-registration was
> not edited. Readers who prefer the letter of the pre-registration should treat
> §6 as not executed as specified.

Observed mean repetition variance: P1 0.0305 dB², P2 0.0196 dB², P3 0.1304 dB². Composed P4 floor: **0.1470 dB²** mean, 12.5292 dB² maximum.

Five fresh C1 instances were trained on the noise-augmented P1–P3 set under the
frozen Strict-DG protocol (architecture, optimizer, loss, batching, shuffling,
seeds 42–46 and per-seed epochs all reused by import from the frozen runtime and
recipe), then evaluated on P4 exactly once.

| | Accuracy | Macro-F1 |
|---|---:|---:|
| noise-augmented C1 | 0.166603 ± 0.026448 | 0.129444 ± 0.034461 |
| frozen Strict-DG C1 (reference only) | 0.156635 ± 0.030516 | 0.112231 ± 0.018956 |
| difference | +0.009968 | +0.017213 |
| chance (7 balanced classes) | 0.142857 | — |

**Reading.** Training under the composed P4 noise floor moves accuracy by
+0.0100 and Macro-F1 by +0.0172. Both shifts are smaller than the
population SD of either arm, and both arms sit within a few points of the
0.1429 chance level for seven balanced classes. Matching the source-domain noise
to the composed target floor therefore does not recover P4 performance, and does
not reproduce the failure either — it changes almost nothing. That is consistent
with §5: if the composition is invalid, the floor it composes is not the P4 floor.

The frozen Strict-DG figures appear here and in the §6 report as the reference
point the augmented retraining is measured against. They are not restated as a
new result.

---

## §7 — Peak-warp prediction and predicted collisions

`Delta_min = 0.170238 GHz`; collision threshold = 1.5 × Delta_min = **0.255357 GHz**, fixed before execution.

**21 of 21 tag pairs were flagged as predicted collisions.**

Because the test flags every pair, it is **not discriminative**, and the hit
counts below are attained trivially. They are reported because the
specification asks for them; they are not evidence.

| Comparison set | TagIDs | Hits | Misses | False alarms | Correct rejections |
|---|---|---:|---:|---:|---:|
| few-shot tracked collapse triple (primary) | [5, 6, 7] | **3/3** | 0 | 18 | 0 |
| Strict-DG zero-recall set (secondary) | [3, 5, 6, 7] | **6/6** | 0 | 15 | 0 |

**Final hit/miss verdict: every pair of TagIDs 5/6/7 is predicted to collide (3/3 hits, 0 misses) — but so is every other pair, so the agreement is vacuous.**

The substantive finding is the threshold derivation itself: only
13 of 252 pair/peak/position combinations (5.16%) are reliably separated
in-domain at all. Across the nine `(er, surface)` conditions of a tag, the
within-tag scatter of a reference-peak position exceeds the between-tag
separation almost everywhere. The four reference peaks do not separate the seven
TagIDs in the *source* domains, so composing their shifts cannot single out any
particular triple in the target domain.

---

## What this does and does not say about the additive-composition hypothesis

The simple additive-composition hypothesis is **not well supported by these
diagnostics**. Strong source-domain interactions in surface × position and
encoding-state × position challenge the additive-composition assumption and
reduce confidence in extrapolating the held P4 cell from source marginals.
These measured factor pairs are proxies: they do not directly establish a
distance-by-angle interaction at P4. The main factorial analysis does not
confirm that interaction.

The noise-floor intervention and peak-warp analysis retain diagnostic value at
their preregistered reduced interpretive weight, with the documented post-hoc
noise-formula deviation. Neither validates the simple composition rule; the
peak analysis also finds weak source-domain separation using the four reference
peaks under its conditional frequency mapping.

It does **not** say that distance and angle are physically unrelated to the P4
failure, that a different composition rule could not work, or that the existing
class-collapse conclusion is wrong. §5 measures interaction in surface × position
and encoding-state × position, which are *proxies* for the distance × angle
interaction that cannot be measured without P4 — a proxy result, not the thing
itself. §6 tests one specific noise-matching intervention under one composition
rule, with a documented post-hoc formula override. §7 rests on a conditional
frequency axis whose provenance the repository records as unresolved and whose
branch status is `BLOCKED`. None of the three can be upgraded into a claim about
the Strict-DG result, and none of them was designed to.

---

## Reproducibility

The three sub-experiments are retained under
`06_failure_mechanism/composition_warp/` with their run scripts, reports,
compact numerical outputs, and generated figures. The analysis is diagnostic
only and does not alter the frozen Strict-DG result.

The noise-floor branch reuses the frozen C1 architecture, seeds, preprocessing
contract, and training recipe. The factor-interaction and peak-warp branches
operate on fixed measurements and declared estimands. The PNG figures are
generated without adding a plotting dependency. Raw measurement data are not
distributed in this public repository.
