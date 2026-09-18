# Preregistered representation-versus-signal protocol

Status before primary probe execution: **PREREGISTERED**.

The scientific question is whether condition-general weakness is already present in measured signals or is introduced or amplified by the frozen C1 encoder. P1-P4 are evaluated independently. The indivisible unit is `TagID x ER x surface x position`, with all 50 repeats grouped.

## Frozen synchronized folds

Test fold 0 uses E0/A1, E1/A2, E2/A3; fold 1 uses E0/A2, E1/A3, E2/A1; fold 2 uses E0/A3, E1/A1, E2/A2. For test fold q, validation is Latin fold `(q+1) mod 3`, and training is Latin fold `(q+2) mod 3`. Thus each split has 21 condition blocks and 1,050 rows per position/fold. The same assignment applies to every representation.

## Frozen representations and readouts

The four families are RAW_SIGNAL_281, FIRST_DIFFERENCE_280, C1_ENCODER_EMBEDDING, and PEAK_DESCRIPTOR. The C1 embedding is the frozen `network.12` output (256 dimensions) from the five source-only checkpoints. Probe scaling is fitted only on the current training rows. The C1 encoder's source preprocessing is checkpoint-bound and is never refit.

PEAK_DESCRIPTOR reuses `crfid.analysis.reference_peaks.strongest_window_peaks` in minimum mode with the preregistered ordered-index windows W0 10-69, W1 75-126, W2 136-182, and W3 192-252. Each window contributes index, amplitude, and missing flag (12 total). Locations are reported primarily as ordered indices. Peak validity before probes: **PEAK_DESCRIPTOR_NOT_VALIDATED**.

The primary probe is one linear layer with softmax classification, Adam (learning rate 0.01, weight decay 0.0001), Xavier-uniform weights, zero bias, full-batch capacity 2048, at most 200 epochs, validation row Macro-F1 selection, and patience 20. Seeds are 42-46 and C1 probe/checkpoint seeds are aligned. COSINE_NCM is the sole secondary readout.

## Label boundary and estimand

Test predictions are generated and hashed before the held labels can be opened. Labels are opened once, after freezing, and metrics are computed once. The primary endpoint is Macro-F1 over the 63 out-of-fold condition-block majority predictions per position and seed. Exact vote ties use the lowest canonical class index. Rows are descriptive, never inferential units.

Primary paired contrasts are C1 minus FIRST_DIFFERENCE, RAW minus C1, and (only if valid) PEAK minus C1. RAW minus FIRST_DIFFERENCE is retained as the preregistered diagnostic needed to assess differencing loss. The angle-associated accuracy contrast is `0.5 * [(P2-P1) + (P4-P3)]`.

Uncertainty uses 10,000 deterministic TagID-stratified paired condition-block bootstrap replicates, retaining representation, position, and aligned seed pairing. No hyperparameter, representation window, fold, threshold, or interpretation rule is changed after results are viewed.
