# Scientific interpretation

Classification: `WEAK_OR_UNRELIABLE_TARGET_CALIBRATION_GAIN`.

No observed budget met the preregistered combined threshold of a +0.05 Macro-F1 point gain with a paired interval excluding zero. The preregistered coverage-dominance rule was not met; condition coverage and raw count cannot be cleanly separated below the 42-block ceiling, and the post-coverage contrast is reported descriptively. TagID 6 had the largest descriptive 0-to-500 class-F1 gain (+0.2710); TagID 1 remained weakest at 500 labels (F1 0.0002).

The calibration curve estimates performance on condition blocks excluded in full from
the corresponding support fold. It therefore avoids within-condition repeated-row
leakage, but every positive point is target-assisted because P4 support labels construct
the prototypes. The historical retrospective and matched within-condition references
are deliberately excluded from recovered-gap calculations.

## Thesis-ready conclusion

Using a frozen P1-P3-trained C1 encoder and a fixed cosine-prototype adapter, the block-disjoint P4 labelled-target calibration curve was classified as `WEAK_OR_UNRELIABLE_TARGET_CALIBRATION_GAIN`. The strict source-only anchor had Macro-F1 0.1122; at 500 target labels the target-assisted Macro-F1 was 0.1023 (paired change -0.0099, 95% CI [-0.0734, 0.0552]). No observed budget met the preregistered combined threshold of a +0.05 Macro-F1 point gain with a paired interval excluding zero. Positive-budget results quantify labelled-target calibration to held P4 condition blocks and are not source-only domain generalisation.
