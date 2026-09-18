# Experiment registry

This registry separates experiments by the information available during training, selection, and evaluation. A higher score obtained with more P4 access answers a different question from Strict-DG and is not treated as a replacement benchmark.

## Core access classes

| Experiment | Source / target access | Primary question | Canonical evidence |
|---|---|---|---|
| Pre-DG grouped | Represented acquisition conditions | Is TagID learnable under familiar/grouped conditions? | `results/canonical_metrics/pre_dg/` |
| Strict-DG C1 | P1-P3 for development; P4 only for final scoring | Does a source-selected recipe transfer to the held joint geometry? | `results/canonical_metrics/strict_dg/` |
| IRM | P1-P3 source-only; P4 final scoring | Does invariant-risk training beat its matched ERM control? | `results/canonical_metrics/irm_invariant_risk/` |
| DANN v2 | P1-P3 source-only; P4 final scoring | Does adversarial position suppression improve held-P4 transfer? | `results/canonical_metrics/dann_v2/` |
| GroupDRO | P1-P3 source-only; P4 final scoring | Does worst-source optimisation improve held-P4 transfer? | `results/canonical_metrics/groupdro_worst_source/` |
| Domain-aware Mixup | P1-P3 source-only; stopped at pairing gate | Is lawful matched cross-position interpolation feasible under the frozen split? | `results/canonical_metrics/domain_aware_mixup/` |

Strict-DG and the source-only interventions do not use P4 labels for fitting, hyperparameter selection, model selection, or checkpoint selection.

## Target-labelled and retrospective studies

| Experiment | P4 access | Scientific role | Canonical evidence |
|---|---|---|---|
| Few-shot P4 | Labelled support/query episodes | Small-label calibration | `results/canonical_metrics/few_shot/` |
| Factor-aware few-shot | Labelled support/query | Support-selection strategy | `results/canonical_metrics/p4_factor_aware_few_shot/` |
| Large calibration curve | Labelled support budgets 0-500 | Label-dose response | `results/canonical_metrics/p4_large_calibration_curve/` |
| Trainable linear readout | Labelled target support | Readout-mismatch diagnostic | `results/canonical_metrics/p4_trainable_linear_readout/` |
| Encoder fine-tuning | Labelled target support | Representation-adaptation diagnostic | `results/canonical_metrics/p4_encoder_finetuning/` |
| Historical target-assisted | Full-P4 outcomes influence readout promotion | Retrospective lineage-specific reference | `results/canonical_metrics/historical_target_assisted/` |
| Matched target-assisted | P4 validation labels influence selection | Controlled target-informed selection diagnostic | `results/canonical_metrics/matched_target_assisted/` |
## Diagnostic studies

| Experiment | What is measured | Claim boundary | Canonical evidence |
|---|---|---|---|
| Representation probe | RAW vs frozen C1 TagID accessibility at P4 | Representation-transfer limitation; not proof of total information loss | `results/canonical_metrics/representation_signal_diagnostic/` |
| Fixed-position grouped validity | Learnability within each position | P4 not uniquely hardest | `results/canonical_metrics/fixed_position_grouped_validity/` |
| Angle-distance factorial | Paired position contrasts | Association within this campaign; not unique causal RF attribution | `results/canonical_metrics/angle_distance_factorial/` |
| Source-selection regret | Source ranking vs post-seal P4 diagnostic | Descriptive support for recorded candidate ranking only | `results/canonical_metrics/source_selection_regret/` |
| Reference peaks | Conditional frequency/peak descriptors | Supporting physical context; descriptor validity gate failed | `results/canonical_metrics/reference_peak/` |
| OpenEMS | Assumption-bounded simulated geometry redesign | Exploratory simulation; no validated hardware rescue | `results/canonical_metrics/openems_redesign/` |

## External-corpus studies

| Experiment | Scope | Claim boundary | Canonical evidence |
|---|---|---|---|
| Data1 portability | Four-class within-Data1 learning | No seven-class label equivalence or Tyndall deployment claim | `results/canonical_metrics/external_data1/` |
| Data1 SSL | Auxiliary self-supervised pretraining with sealed Paper4 P4 evaluation | No reliable benefit for the tested treatments | `results/canonical_metrics/external_data1_ssl/` |

## Key numerical anchors

- Strict-DG C1: Accuracy **0.156635**, Macro-F1 **0.112231** over seeds 42-46.
- IRM matched effect: **-0.019330**, 95% **[-0.066939, 0.025387]**.
- DANN v2 matched effect: **+0.016037**, 95% **[-0.035578, 0.068624]**.
- GroupDRO matched effect: **+0.001405**, 95% **[-0.042978, 0.039489]**.
- Encoder-clean P4 probe: RAW minus C1 Macro-F1 **+0.0951 [0.0173, 0.1747]**.
- Matched target-informed selection A1-A0: Macro-F1 **+0.002304 [-0.001057, 0.006679]**.
- OpenEMS best confirmed minimum-separation change: approximately **0.681%**, below the 5% gate.

The public result directories contain the compact evidence required to interpret these claims. Full historical replay bundles are not distributed; see REPRODUCIBILITY.md for input requirements.
