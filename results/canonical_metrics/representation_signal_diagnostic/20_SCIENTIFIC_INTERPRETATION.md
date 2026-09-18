# Scientific interpretation

Main classification: **`REPRESENTATION_DEPENDENT_MIXED_RESULT`**.

Reliable representation effects were position- or contrast-dependent, so neither a single signal-level nor encoder-level explanation is supported.

## Required explicit findings

- `RAW_SIGNAL_281`: PASS; passing positions P1, P2, P3, P4 across all five seeded controls.
- `FIRST_DIFFERENCE_280`: PASS; passing positions P1, P2, P3, P4 across all five seeded controls.
- `C1_ENCODER_EMBEDDING`: PASS; passing positions P1, P2, P3, P4 across all five seeded controls.
- `PEAK_DESCRIPTOR`: learning validity not run because peak extraction was `PEAK_DESCRIPTOR_NOT_VALIDATED`.

- Peak extractor validity: **PEAK_DESCRIPTOR_NOT_VALIDATED**. Global missing frequency was 0.117004; all deterministic synthetic controls passed.
- Did raw signal outperform the learned embedding? Yes, under the preregistered primary block-only analysis at P4; seed sensitivity is reported below.
- Did first difference outperform the learned embedding? Yes, under the preregistered primary block-only analysis at P4; seed sensitivity is reported below.
- Did peak descriptors rescue condition-general performance? Not assessed scientifically because PEAK_DESCRIPTOR_NOT_VALIDATED.
- Was angle-associated degradation already present before encoding? Yes; the RAW_SIGNAL_281 angle-associated interval was wholly below zero.
- Signal-level versus encoder-level conclusion: **`REPRESENTATION_DEPENDENT_MIXED_RESULT`**; no stronger causal claim is made.

## Five-seed pooled 63-block estimands

| Position | Representation | Block Macro-F1 | Block Accuracy |
|---|---|---:|---:|
| P1 | RAW_SIGNAL_281 | 0.5192 | 0.5143 |
| P1 | FIRST_DIFFERENCE_280 | 0.3814 | 0.4000 |
| P1 | C1_ENCODER_EMBEDDING | 0.7632 | 0.7651 |
| P2 | RAW_SIGNAL_281 | 0.2737 | 0.2825 |
| P2 | FIRST_DIFFERENCE_280 | 0.2481 | 0.2667 |
| P2 | C1_ENCODER_EMBEDDING | 0.7623 | 0.7714 |
| P3 | RAW_SIGNAL_281 | 0.5363 | 0.5492 |
| P3 | FIRST_DIFFERENCE_280 | 0.3561 | 0.3714 |
| P3 | C1_ENCODER_EMBEDDING | 0.9095 | 0.9143 |
| P4 | RAW_SIGNAL_281 | 0.2427 | 0.2571 |
| P4 | FIRST_DIFFERENCE_280 | 0.2359 | 0.2540 |
| P4 | C1_ENCODER_EMBEDDING | 0.1472 | 0.1524 |

## Paired representation uncertainty

RAW Macro-F1 **0.2427** versus C1 **0.1472**, an observed mean contrast of **+0.0955**. The preregistered primary paired TagID-stratified block bootstrap gives an estimate of **+0.0951**, 95% CI **[0.0173, 0.1747]**. A block-plus-probe-seed sensitivity interval crosses zero, **[-0.0015, 0.1891]**. The primary evidence supports reduced linearly accessible TagID discrimination in C1 at P4, while seed-level uncertainty weakens the strength of that inference; it does not establish loss of all TagID information.

The estimates in the table below are bootstrap-distribution means, not the observed contrasts in `13_REPRESENTATION_CONTRASTS.csv`. The observed RAW-minus-C1 contrast is 0.0955360313; its primary bootstrap estimate is 0.0950544281.

| Position | Paired block Macro-F1 contrast | Estimate | 95% interval |
|---|---|---:|---:|
| P1 | C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280 | 0.3841 | [0.2688, 0.4925] |
| P1 | RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING | -0.2472 | [-0.3709, -0.1201] |
| P1 | RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280 | 0.1370 | [0.0429, 0.2309] |
| P2 | C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280 | 0.5165 | [0.4261, 0.6061] |
| P2 | RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING | -0.4923 | [-0.5869, -0.3950] |
| P2 | RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280 | 0.0241 | [-0.0731, 0.1216] |
| P3 | C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280 | 0.5587 | [0.4603, 0.6499] |
| P3 | RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING | -0.3784 | [-0.4824, -0.2723] |
| P3 | RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280 | 0.1803 | [0.0809, 0.2778] |
| P4 | C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280 | -0.0876 | [-0.1543, -0.0236] |
| P4 | RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING | 0.0951 | [0.0173, 0.1747] |
| P4 | RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280 | 0.0075 | [-0.0744, 0.0888] |

## Angle-associated secondary analysis

| Representation | Angle-associated block-accuracy contrast | 95% interval |
|---|---:|---:|
| RAW_SIGNAL_281 | -0.2619 | [-0.3587, -0.1619] |
| FIRST_DIFFERENCE_280 | -0.1254 | [-0.1952, -0.0540] |
| C1_ENCODER_EMBEDDING | -0.3778 | [-0.4381, -0.3159] |

All intervals use 10,000 deterministic TagID-stratified condition-block bootstrap replicates. Full numerical sensitivity results are in `18_BOOTSTRAP_AND_SEED_SENSITIVITY.csv`; COSINE_NCM remains secondary in `19_NCM_SECONDARY_RESULTS.csv`.
