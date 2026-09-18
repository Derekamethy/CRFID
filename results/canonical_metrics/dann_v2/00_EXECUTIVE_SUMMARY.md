# DANN v2 executive summary

Classification: `DANN_V2_NO_RELIABLE_POSITION_SUPPRESSION_OR_P4_BENEFIT`.

Implementation validity passed, including forward identity, sign and magnitude matching, lambda-zero equivalence, fold-local labels, and development/final schedule regression. Development completed **165/165** source-only runs.

The source-only selector chose **lambda = 0.1** from the weak position-reduction candidate set. Relative to matched A0, familiar validation changed by **+0.000705**, the position-probe point estimate changed by **-0.006807** with 95% interval **[-0.019896, +0.006788]**, and the TagID probe changed by **-0.007686**. Because the position-probe interval crosses zero, the study does **not** show reliable suppression of source-position information.

The source-mechanism continuation gate passed and allowed the frozen P4 stage to proceed. Historical sealed artifacts retain the old label `PASS_SELECTIVE_INVARIANCE_GATE`; that label is preserved only as protocol history and is not interpreted as inferential confirmation of invariance.

P4 completed **15/15** runs. Mean block Accuracy / Macro-F1 were A0 **0.133333 / 0.101116**, A1 **0.111111 / 0.104733**, and A2 **0.139683 / 0.117153**. The matched A2-A0 Macro-F1 effect was **+0.016037** with block-plus-training-seed 95% interval **[-0.035578, +0.068624]**. The interval crosses zero, so there is no reliable P4 benefit.

DANN v2 was designed after earlier P4 outcomes were historically known. Its model development and selection use P1-P3 only, with final P4 predictions frozen before label access; it is therefore a governed post-hoc source-only-developed follow-up rather than pristine never-before-seen target confirmation.
