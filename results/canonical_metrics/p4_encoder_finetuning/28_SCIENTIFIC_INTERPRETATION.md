# Scientific interpretation

## Final classification

`ENCODER_ADAPTATION_DOES_NOT_RESCUE_HELD_CONDITION_TRANSFER`

At 500 labels, frozen-linear Macro-F1 was 0.1427, Partial was 0.1319, and Full was 0.1296. The paired changes were -0.0109 for Partial-minus-Frozen, -0.0131 for Full-minus-Frozen, and -0.0022 for Full-minus-Partial.

The classification follows the preregistered +0.05 practical threshold and paired 95% interval rule. Support performance, class-specific changes, condition strata, representation displacement, and source retention are diagnostic only and did not select a model.

This closes the current P4 labelled-target adaptation model line. No additional architecture or adaptation family was launched.
