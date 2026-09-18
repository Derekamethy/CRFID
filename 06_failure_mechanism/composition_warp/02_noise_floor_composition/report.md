# Section 6 — Composite P4 noise floor and noise-augmented retraining

Sub-module: `06_failure_mechanism/composition_warp/02_noise_floor_composition`.

## Question

If the P4 failure were driven by the *measurement noise* that the 150 mm / 45°
condition inherits from its two factor levels, then raising the source-domain
noise to the composed P4 level should reproduce part of that failure — and
training under it should confer some robustness. This sub-experiment composes
the noise floor from P1/P2/P3 alone, retrains under it, and reads P4 once.

## Stage 1 — noise floor

Per-condition repetition variance `sigma2[c, p](f)` is computed across the 50
repetitions of each of the 63 `(tag_id, er, surface)` conditions at each source
position, with `ddof = 1`, over all 281 ordered signal positions.

Pre-registered primary rule: `sigma2_P2 + sigma2_P3 - sigma2_P1`.
Pre-declared fallback (if more than 1 % of cells go negative):
`sigma2_P2 * sigma2_P3 / sigma2_P1`, additive in `log sigma^2`.

**Negative cells under the primary rule: 4860 of 17703 (0.274530).**

**Formula used: `additive_on_variance` — `sigma2_P2 + sigma2_P3 - sigma2_P1, negatives clipped to zero`.**

The pre-registered automatic switch fired: additive-on-variance produced negative estimates in 0.274530 of cells, above the pre-declared 0.01 limit. The pre-declared log-variance additive fallback was then computed and rejected on a recorded numerical-validity criterion: its maximum estimate is 1.91408e+07 dB^2 against a total measured source signal variance of 46.6932 dB^2, because sigma2_P1 reaches values far below the typical repetition variance and the ratio estimator diverges. An estimated noise floor larger than the entire dynamic range of the data is not a noise floor. The primary additive-on-variance rule was therefore used, with its 4860 negative cells clipped to zero, which is interpretable as 'the composed P4 floor does not exceed the observed floor at this point'. This is a documented post-hoc deviation from the pre-registration, not a silent substitution; both estimators are reported.

> **Documented deviation from pre-registration.** The pre-registered automatic
> switch selected `log_variance_additive`. That estimator was computed and then
> rejected on a recorded numerical-validity criterion, and
> `additive_on_variance` was used instead. The criterion is:
> *max estimated noise variance must not exceed the total variance of the measured source signal, and the estimate must be finite everywhere*. It was formulated after the Stage-1 diagnostics and is
> therefore post-hoc. It is recorded here rather than applied silently, and both
> estimators are reported in `results/noise_floor_estimates.json`.
>
> | Estimator | Max (dB²) | Min (dB²) | Negative cells | Non-finite cells |
> |---|---:|---:|---:|---:|
> | `additive_on_variance` | 12.5292 | -10.7555 | 4860 | 0 |
> | `log_variance_additive` | 1.91408e+07 | 1.30367e-14 | 0 | 0 |
> | *total measured source signal variance* | 46.6932 | — | — | — |
>
> The rejected ratio estimator exceeds the total variance of the measured signal by
> a factor of about 4.1e+05,
> because `sigma2_P1` reaches values far below the typical repetition variance. A noise
> floor larger than the entire dynamic range of the data cannot be a noise floor.

| Quantity | Mean (dB²) | Median (dB²) | Max (dB²) |
|---|---:|---:|---:|
| observed `sigma2` at P1 | 0.030476 | 0.000005 | 10.756914 |
| observed `sigma2` at P2 | 0.019562 | 0.000003 | 6.352342 |
| observed `sigma2` at P3 | 0.130354 | 0.000039 | 10.889735 |
| **composed `sigmahat2` for P4** | **0.147014** | **0.000099** | **12.529174** |

The composed P4 floor is 4.8239× the mean P1 variance, 7.5152× P2 and 1.1278× P3.

## Stage 2 — augmented retraining

Independent zero-mean Gaussian noise of variance
`max(0, sigmahat2_P4[c](f) - sigma2[c,p](f))` is added per sample and per signal
position, so each augmented condition's repetition variance matches the composed
P4 floor. Noise is applied to the **P1–P3 training data only**; P4 is never
augmented and preprocessing is fitted on augmented source data only. The noise
generator seed is `1000000 * training_seed + 7`, making every draw reproducible.

Everything else is reused by import from the frozen Strict-DG runtime — the
`NeutralSourceOnlyCNN1D` architecture (142,855 parameters), `_optimizer`,
`_criterion`, `_erm_train_epoch`, and the canonical degenerate-scale policy —
with hyperparameters read from `$CRFID_COMPOSITION_FROZEN_STRICT_DG/recipe.json`:
AdamW lr 1e-3 / wd 1e-4, batch 256, unweighted cross-entropy, seeds 42–46 at
13/13/9/10/9 epochs. Five fresh models were trained. Nothing in the frozen
Strict-DG release was written to.

Final-epoch training loss per seed: seed 42 `0.756347`, seed 43 `0.811847`, seed 44 `1.142290`, seed 45 `1.028346`, seed 46 `1.109839`.

## Stage 3 — the single held-out P4 evaluation

Target access was machine-enforced through the existing gate. This sub-module
built its own frozen release (recipe, per-seed preprocessing states, five
verified checkpoints, release manifest, authorization) and minted a
`TargetAccessToken` via `crfid.governance.strict_target_authorization.authorize_target_access`
against **its own** release directory. P4 was read from disk exactly
3 times (once per surface file) through the existing
`load_p4_once`. Predictions were serialized before labels were released for
scoring. Target training steps: 0. Target preprocessing refits: 0.

| Seed | Epochs | Accuracy | Macro-F1 | Worst-class recall | Zero-recall classes |
|---:|---:|---:|---:|---:|---:|
| 42 | 13 | 0.174286 | 0.136084 | 0.000000 | 2 |
| 43 | 13 | 0.121587 | 0.075707 | 0.000000 | 4 |
| 44 | 9 | 0.190159 | 0.151984 | 0.000000 | 3 |
| 45 | 10 | 0.193016 | 0.174863 | 0.000000 | 2 |
| 46 | 9 | 0.153968 | 0.108581 | 0.000000 | 3 |

**Accuracy 0.166603 ± 0.026448** (population SD, five seeds, equal weight).

**Macro-F1 0.129444 ± 0.034461.**

### Reference point

Quoted inside this sub-experiment only, as the reference point the augmented retraining is measured against. It is not restated as a new result and is not inserted into any existing comparison table.

| | Accuracy | Macro-F1 |
|---|---:|---:|
| noise-augmented C1 (this sub-experiment) | 0.166603 ± 0.026448 | 0.129444 ± 0.034461 |
| frozen Strict-DG C1 | 0.156635 ± 0.030516 | 0.112231 ± 0.018956 |
| difference | +0.009968 | +0.017213 |
| chance (7 balanced classes) | 0.142857 | — |

## Outputs

* `results/noise_floor_estimates.json` — variance definitions, the composition rule actually used
  and why, observed and composed variance summaries, the condition-averaged 281-point curves,
  and the augmentation rule.
* `results/retrain_eval_summary.json` — per-seed and aggregate P4 metrics, the target-access record
  including the minted token and the P4 file digests, and the frozen Strict-DG reference.
* `results/p4_evaluation_declaration.json` — the evaluation declaration, written before the first
  numerical P4 load.
* `execution/` — the sub-module's own frozen release, P4 custody manifest and per-seed prediction
  bundles. Excluded from the review package: checkpoints are `.pt` and the bundles carry P4 arrays.

## Scope

Diagnostic and mechanistic only. This does not modify or challenge the frozen
Strict-DG, Few-Shot or External-Data1 results, and these numbers are not
inserted into any existing comparison table.
