# Pre-registration — `composition_warp` sub-module

Status: **WRITTEN BEFORE ANY EXPERIMENT WAS RUN.** No result file in this
sub-module existed when this document was committed to disk. Every threshold
below is fixed at the moment of writing and is not revised afterwards.

Branch scope: `06_failure_mechanism/composition_warp/` only. Diagnostic /
mechanistic study. Not a performance claim.

---

## 0. Public input mapping

The public version keeps governed measurement arrays and frozen runtime artifacts
outside Git. The diagnostic uses the following explicit bindings.

| Input | Public binding | Role |
|---|---|---|
| source inputs | `$CRFID_COMPOSITION_SOURCE_INPUTS` + `src/crfid/strict_runtime/` | external canonical P1–P3 arrays, registry, splits, preprocessing |
| `02_strict_dg/` | `configs/strict_dg/canonical.yaml`, `$CRFID_COMPOSITION_FROZEN_STRICT_DG/recipe.json`, `scripts/train_and_freeze_strict_dg.py`, `src/crfid/strict_runtime/{neutral_model,phase3b_execution,target_evaluation,scale_policy}.py`, `src/crfid/governance/strict_target_authorization.py` | frozen C1 definition, training config, training + evaluation protocol |
| reference peak evidence | `src/crfid/analysis/reference_peaks.py`, `results/canonical_metrics/reference_peak/conditional_frequency_axis_281_5_to_8.csv`, `results/canonical_metrics/reference_peak/FREQUENCY_VALUE_MAP_REPORT.md` | peak extraction code, conditional 281-point axis, verified Lc/L3/L2/L1 windows |
| P4 class-collapse evidence | historical few-shot and Strict-DG collapse summaries retained in this diagnostic | comparison-only conclusion text; no raw P4 data |

### Confirmed environment

`crfid-env-check` returns `PASS_CRFID_LOCKED_ENVIRONMENT`. Locked identities:
CPython 3.12.13, torch 2.12.0+cpu, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.0, pandas 3.0.3, PyYAML 6.0.3, pytest 9.1.1. These match the
specification exactly. All execution in this sub-module uses `crfid-python`.

### Confirmed dataset shape

From `CANONICAL_SOURCE_REGISTRY.csv` + `source_signals_float64.npy`
(read-only): 9,450 source rows of 281 float64 points, fully crossed as
7 `tag_id` × 3 `er` × 3 `surface` × 3 `position` (P1/P2/P3) × 50 `repeat_index`.
All 189 `raw_condition_id` blocks contain exactly 50 repetitions. P4 adds
3 × 1,050 = 3,150 rows, giving 12,600 in total. Values are dB
(observed range −73.911 to −30.696).

**Repetition-level pairing (needed for §6, and for the paired form of the §3
hypothesis):** the key `(tag_id, er, surface, repeat_index)` has exactly 3,150
groups of exactly 3 members — one per source position. P1/P2/P3 are therefore
fully paired at repetition level, and per-condition variance across repetitions
is computable within each position from the 50-member `raw_condition_id` blocks.

### Position factor definition

P1 = 50 mm / 0°, P2 = 50 mm / 45°, P3 = 150 mm / 0°, P4 = 150 mm / 45°. This is
the 2 × 2 distance × angle design. Both P4 factor levels (150 mm, 45°) are
independently observed in the source domains — 150 mm in P3, 45° in P2.

---

## 1. Hypothesis

**Additive / no-interaction model over the 2 × 2 distance × angle factor
design.** For each TagID and each condition `c = (tag_id, er, surface)`, and at
each of the 281 ordered signal positions `f`:

```
mu_hat[c, P4](f) = mu[c, P2](f) + mu[c, P3](f) - mu[c, P1](f)
```

where `mu[c, p](f)` is the mean over the 50 repetitions of condition `c` at
source position `p`. Equivalently: the effect of changing distance and the
effect of changing angle combine additively, with no distance × angle
interaction term. The construction uses paired P1/P2/P3 samples only.

The same additive form is carried into §6 (composition of the noise floor) and
§7 (composition of resonance-peak shifts).

## 2. Target-domain information boundary

Only the **factor-level definitions** of P4 are used: the labels "150 mm" and
"45°". Both levels are independently observed in the source domains. No P4
measurement, no P4 label, and no P4 summary statistic is read anywhere in this
sub-module, with exactly two declared exceptions:

1. **§6, the single held-out P4 evaluation.** One evaluation pass, after the
   five noise-augmented models are trained on P1–P3 and frozen. It mirrors the
   existing Strict-DG evaluation protocol: this sub-module builds its own frozen
   release (recipe, preprocessing state, five checkpoints, release manifest,
   target-access authorization), mints a `TargetAccessToken` through the
   existing `crfid.governance.strict_target_authorization.authorize_target_access`
   against **its own** frozen release directory, and reads P4 through the
   existing `crfid.strict_runtime.target_evaluation.load_p4_once`. Predictions
   are serialized before target labels are released for scoring, exactly as in
   `scripts/evaluate_frozen_p4.py`. The frozen Strict-DG release is read for
   digest comparison only and is never modified, and its own token is never
   minted for this sub-module's models.
2. **§7, the comparison-only read of the existing `class_collapse`
   conclusion.** Conclusion text only (which TagIDs collapse). No raw P4 data,
   no P4 spectra, no P4 embeddings.

No other P4 access occurs. §5 and §7 are computed from P1–P3 alone.

## 3. Fixed thresholds — chosen and recorded before any result exists

### §5 — interaction / main-effect ratio

For a factor pair `A × B` with cell means `mu[a, b](f)` over the 281 ordered
positions, define the two-way decomposition

```
grand(f)        = mean over all cells
alpha[a](f)     = rowmean[a](f) - grand(f)
beta[b](f)      = colmean[b](f) - grand(f)
additive[a,b](f)= grand(f) + alpha[a](f) + beta[b](f)
gamma[a,b](f)   = mu[a,b](f) - additive[a,b](f)          # interaction
```

with summary scalars

```
RMS_interaction = sqrt( mean over (a, b, f) of gamma^2 )
RMS_main_A      = sqrt( mean over (a, f)    of alpha^2 )
RMS_main_B      = sqrt( mean over (b, f)    of beta^2  )
ratio           = RMS_interaction / min(RMS_main_A, RMS_main_B)
```

**Threshold: `ratio > 0.30` ⇒ `HIGH`. Otherwise `LOW`.**

Rationale for 0.30: the additive hypothesis is used in §6/§7 as a *point
predictor*, not as a statistical model to be accepted. An interaction whose RMS
is under 30 % of the weaker of the two main effects leaves the additive
prediction dominated by the main effects it is built from. Above 30 % the
residual is no longer a second-order correction and the extrapolation to an
unobserved cell is not defensible on this evidence.

Reported per factor pair, in dB, for the two fully crossed pairs present in the
source data: **surface × position** and **encoding-state (`er`) × position**.
A per-TagID breakdown of the same two decompositions is reported as a secondary
result; the pooled value is the one the threshold is applied to.

### §7 — predicted-collision threshold

Define, using source data only:

* Per source position `p` and peak name `k`, the per-TagID mean peak frequency
  `mu[t, p, k]` is the mean over that tag's 9 `(er, surface)` conditions.
* Tag pair `(i, j)` is **reliably separated in-domain** at `(p, k)` if
  `|mu[i,p,k] - mu[j,p,k]| > s_pooled(i, j, p, k)`, where `s_pooled` is the
  pooled within-tag standard deviation of that peak frequency across the same
  9 conditions.
* `Delta_min` = the **minimum** `|mu[i,p,k] - mu[j,p,k]|` over all reliably
  separated `(i, j, p, k)`. This is the empirical minimum reliably-separated
  peak spacing observed in-domain.

**Threshold: a tag pair is flagged as a `predicted collision` when its predicted
P4 separation is `< 1.5 * Delta_min`.**

Applied per peak name to the predicted per-TagID P4 peak positions of all seven
TagIDs. A pair is a *predicted collision overall* if it is flagged on at least
one of the four peaks.

### §6 — variance-composition rule (declared in advance, not a threshold)

Primary formula, the exact variance analogue of the §1 hypothesis:

```
sigmahat2[c, P4](f) = sigma2[c,P1](f) + (sigma2[c,P3](f) - sigma2[c,P1](f))
                                      + (sigma2[c,P2](f) - sigma2[c,P1](f))
                    = sigma2[c,P2](f) + sigma2[c,P3](f) - sigma2[c,P1](f)
```

Variance is computed across the 50 repetitions within each condition block,
`ddof = 1`.

Additive-on-variance is not guaranteed non-negative. **Declared switching rule,
fixed now:** if more than 1 % of the `(c, f)` cells give a negative primary
estimate, the sub-experiment switches to the log-variance additive composition

```
sigmahat2[c, P4](f) = sigma2[c,P2](f) * sigma2[c,P3](f) / sigma2[c,P1](f)
```

which is additive in `log sigma^2`, non-negative by construction, and is the
natural composition rule for a positive scale parameter. If 1 % or fewer cells
are negative, the primary formula is kept and negatives are clipped to zero.
Whichever branch is taken, the negative-cell fraction and the formula actually
used are reported.

**Augmentation rule, fixed now.** Training samples of condition `c` at source
position `p` already carry variance `sigma2[c,p](f)`. Independent zero-mean
Gaussian noise of variance

```
delta2[c, p](f) = max(0, sigmahat2[c,P4](f) - sigma2[c,p](f))
```

is added per sample, so the augmented per-condition variance matches the
estimated P4 noise floor. Noise is drawn from a `numpy.random.Generator` seeded
as `1_000_000 * seed + 7` for training seed `seed`, making every draw
deterministic and reproducible. Noise is added to the **training data only**;
P4 is never augmented, and preprocessing is fitted on the augmented source data
only.

**Training protocol, fixed now.** Reused by import from the frozen Strict-DG
path, not copy-pasted: `crfid.strict_runtime.neutral_model.initialize_model`
(the `NeutralSourceOnlyCNN1D`, 142,855 parameters),
`crfid.strict_runtime.phase3b_execution.{_optimizer, _criterion, _erm_train_epoch}`,
`crfid.strict_runtime.scale_policy.apply_scale_policy` with `CANONICAL_MODE`.
Hyperparameters are read from `$CRFID_COMPOSITION_FROZEN_STRICT_DG/recipe.json`:
candidate `C1_FIRST_DIFFERENCE_ERM_1DCNN`, first-difference representation,
AdamW lr 1e-3 / wd 1e-4, batch 256, unweighted cross-entropy, seeds
42–46, per-seed epochs 13/13/9/10/9, `fold_id = "ALL_P1_P2_P3"`,
`stage = "outer_refit"`. Nothing in the frozen release is written to.

The frozen Strict-DG C1 P4 numbers (Accuracy 0.15663492063492063, Macro-F1
0.11223088363457531) are quoted **inside `02_noise_floor_composition/report.md`
only**, as a reference point. They are not restated as a new result and this
sub-module's numbers are not inserted into any existing comparison table.

## 4. Stopping rule

If §5 returns `HIGH` for either factor pair, §6 and §7 are **still run in full**.
Their interpretive weight is **explicitly downgraded** in
`composition_warp_summary.md`. §6 and §7 are not skipped, and the §5 threshold
of 0.30 is not adjusted after seeing the §5 result.

## 5. Declaration of scope

Results from this sub-module are **diagnostic / mechanistic only**. They must
not be inserted into the Strict-DG, Few-Shot, or External-Data1 comparison
tables. This sub-module does not modify, retrain, re-audit, or challenge any
frozen result in any other branch. Nothing outside
`06_failure_mechanism/composition_warp/` is written to.

A physical frequency axis is **not asserted**. The Strict-DG canonical
configuration records `frequency_axis_status:
PHYSICAL_FREQUENCY_AXIS_PROVENANCE_UNRESOLVED`, and the reference-peak branch
release status is `BLOCKED` because the primary paper states 700 upstream points
with no preserved 700→281 transformation. §7 therefore uses the *conditional*
281-point 5–8 GHz axis, and every frequency in GHz reported by §7 is conditional
on that unresolved axis. §5 and §6 use ordered signal positions and do not
depend on the axis at all.
