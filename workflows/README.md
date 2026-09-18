# Workflow index

Workflows are grouped by scientific role. Directory numbers are identifiers, not execution order.

Publicly supported commands are the Strict-DG compact-evidence verifier, the prepared-array validator, and the angle-distance preflight/execution path with governed measurements. Other entries below index retained implementations and require their documented frozen inputs or omitted archives; listing a directory does not imply raw-data-only reproducibility. See [reproducibility](../REPRODUCIBILITY.md).

## Core benchmark

| Workflow | Purpose |
|---|---|
| [`00_prepare_data/`](00_prepare_data/) | Prepare governed dataset inputs |
| [`01_pre_dg_baseline/`](01_pre_dg_baseline/) | Familiar-condition / Pre-DG baseline |
| [`02_strict_source_only_dg/`](02_strict_source_only_dg/) | Public retained-evidence verification; historical source-only training implementation |
| [`19_dann_v2/`](19_dann_v2/) | Matched DANN v2 source-only DG benchmark |
| [`15_groupdro_worst_source/`](15_groupdro_worst_source/) | Matched GroupDRO benchmark |
| [`16_irm_invariant_risk/`](16_irm_invariant_risk/) | Matched IRMv1 benchmark |
| [`17_domain_aware_mixup/`](17_domain_aware_mixup/) | Domain-aware Mixup feasibility study |

## Diagnostics and target-labelled studies

| Workflow | Purpose |
|---|---|
| [`04_failure_mechanism_analysis/`](04_failure_mechanism_analysis/) | Signal, representation, and failure-mechanism diagnostics |
| [`09_fixed_position_grouped_validity/`](09_fixed_position_grouped_validity/) | Within-position grouped learnability |
| [`10_angle_distance_factorial_contrast/`](10_angle_distance_factorial_contrast/) | Angle/distance physical-factor diagnostic |
| [`12_representation_signal_diagnostic/`](12_representation_signal_diagnostic/) | Raw signal vs learned-representation comparison |
| [`13_source_selection_regret/`](13_source_selection_regret/) | Source-only model-selection validity |
| [`14_dann_position_intervention/`](14_dann_position_intervention/) | Controlled position-suppression diagnostic |
| [`05_few_shot_p4_adaptation/`](05_few_shot_p4_adaptation/) | Historical few-shot target calibration |
| [`11_p4_factor_aware_few_shot/`](11_p4_factor_aware_few_shot/) | Factor-aware support selection |
| [`12_p4_large_calibration_curve/`](12_p4_large_calibration_curve/) | P4 label-budget calibration curve |
| [`13_p4_trainable_linear_readout/`](13_p4_trainable_linear_readout/) | Trainable target readout |
| [`14_p4_encoder_finetuning/`](14_p4_encoder_finetuning/) | Historical partial/full encoder fine-tuning; omitted scientific archive required |

## Supporting branches

| Workflow | Purpose |
|---|---|
| [`03_target_assisted_adaptation/`](03_target_assisted_adaptation/) | Retrospective target-informed reference |
| [`05_external_data1_validation/`](05_external_data1_validation/) | External Data1 validation |
| [`08_external_data1_ssl_strict_dg/`](08_external_data1_ssl_strict_dg/) | Auxiliary-data SSL under sealed P4 evaluation |
| [`07_openems_physical_redesign/`](07_openems_physical_redesign/) | Exploratory OpenEMS redesign study |

The information-access regimes differ. Use [`../EXPERIMENT_REGISTRY.md`](../EXPERIMENT_REGISTRY.md) before comparing results across workflows.