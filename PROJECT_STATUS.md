# Project status

This file summarises the public experiment set. Status labels describe the **scientific claim boundary**, not whether an experiment executed successfully.

| Experiment | Information regime | Status | Public evidence | Final conclusion |
|---|---|---|---|---|
| Pre-DG grouped baseline | Familiar/grouped conditions | CONTEXT | `results/canonical_metrics/pre_dg/` | Familiar-condition learning can be strong; it is not unseen-geometry transfer. |
| Strict-DG C1 | P1-P3 source-only; P4 final | FROZEN | `results/canonical_metrics/strict_dg/` | Accuracy 0.156635; Macro-F1 0.112231; severe held-P4 failure. |
| CORAL source candidate | P1-P3 source-only selection | CONTEXT | `results/canonical_metrics/strict_dg/` | Source-side candidate only; no canonical P4 performance claim. |
| Few-shot P4 | Labelled P4 support/query | TARGET-LABELLED | `results/canonical_metrics/few_shot/` | Small episode-level calibration gains; not source-only DG. |
| IRM | Source-only, matched ERM control | FROZEN | `results/canonical_metrics/irm_invariant_risk/` | P4 Macro-F1 effect -0.019330; interval crosses zero. |
| DANN v2 | Source-only, matched ERM control | FROZEN | `results/canonical_metrics/dann_v2/` | P4 Macro-F1 effect +0.016037; interval crosses zero; position suppression not reliable. |
| GroupDRO | Source-only, matched ERM control | FROZEN | `results/canonical_metrics/groupdro_worst_source/` | P4 Macro-F1 effect +0.001405; interval crosses zero. |
| Domain-aware Mixup | Source-only feasibility gate | FEASIBILITY | `results/canonical_metrics/domain_aware_mixup/` | 50% lawful-parent coverage; training stopped before P4 evaluation. |
| Unified DG benchmark | Synthesis of matched effects | FROZEN | `results/canonical_metrics/unified_dg_benchmark/` | None of IRM, DANN v2, or GroupDRO shows a reliable matched P4 benefit. |
| Factor-aware few-shot | Labelled P4 diagnostic | DIAGNOSTIC | `results/canonical_metrics/p4_factor_aware_few_shot/` | No reliable advantage over random support at matched budget. |
| Large P4 calibration | Labelled P4 budgets 0-500 | DIAGNOSTIC | `results/canonical_metrics/p4_large_calibration_curve/` | Non-monotonic; more labels do not give a simple prototype dose-response. |
| Trainable P4 readout | Labelled P4 support | DIAGNOSTIC | `results/canonical_metrics/p4_trainable_linear_readout/` | Macro-F1 0.142740 at 500 labels; modest and uncertain recovery. |
| Encoder fine-tuning | Labelled P4 support | DIAGNOSTIC | `results/canonical_metrics/p4_encoder_finetuning/` | Partial/full fine-tuning does not improve on the frozen linear readout. |
| Historical target-assisted | Full-P4-informed retrospective selection | HISTORICAL | `results/canonical_metrics/historical_target_assisted/` | Historical carrier reaches 0.577143 Accuracy / 0.583569 Macro-F1; not canonical-C1 recoverability or an independent target test. |
| Matched target-assisted | P4 validation-informed selection | DIAGNOSTIC | `results/canonical_metrics/matched_target_assisted/` | A1-A0 Macro-F1 +0.002304, CI [-0.001057, 0.006679]; no reliable selector gain. |
| Representation diagnostic | P4 diagnostic probes | DIAGNOSTIC | `results/canonical_metrics/representation_signal_diagnostic/` | RAW 0.2427 vs C1 0.1472 Macro-F1; representation-transfer limitation supported. |
| Fixed-position grouped validity | Within-position grouped evaluation | DIAGNOSTIC | `results/canonical_metrics/fixed_position_grouped_validity/` | P4 is not uniquely hardest; P2 is comparably weak. |
| Angle-distance factorial | Paired physical diagnostic | DIAGNOSTIC | `results/canonical_metrics/angle_distance_factorial/` | Adverse 45-degree association supported; global distance and interaction not confirmed. |
| Reference-peak analysis | Frequency-domain supporting evidence | DIAGNOSTIC | `results/canonical_metrics/reference_peak/` | Descriptor validity gate failed; P4 missingness is lowest and does not explain collapse. |
| External Data1 | Within-external-corpus portability | SUPPORTING | `results/canonical_metrics/external_data1/` | Pipeline portability supported within four-class Data1 only. |
| External Data1 SSL | Auxiliary-data source-only treatment | SUPPORTING | `results/canonical_metrics/external_data1_ssl/` | No reliable SSL benefit over matched supervised-only control. |
| OpenEMS redesign | Exploratory simulation | DIAGNOSTIC | `results/canonical_metrics/openems_redesign/` | NOT_CONFIRMED; 0.681% best confirmed change, below 5% gate; no hardware validation. |

For protocol details and target-access boundaries, see [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md).
