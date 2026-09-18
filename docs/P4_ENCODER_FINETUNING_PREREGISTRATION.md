# P4 Encoder Fine-Tuning Diagnostic v1: Preregistration

The normative preregistration is
`configs/p4_encoder_finetuning/preregistration.json`. Its SHA-256 sidecar is the
execution seal. This document was written before source-only selection outcomes,
new P4 fine-tuning predictions, or new P4 query metrics were produced.

The primary intervention is limited to two independently initialized supervised
adaptation arms. Partial updates the final Conv1d/GroupNorm representation block
and linear head. Full updates the entire C1 encoder and linear head. Both begin
from the same original source checkpoint and the exactly reproduced parent
support-fitted linear head; neither begins from the other arm.

Hyperparameters are selected separately for Partial and Full using only the
preregistered P1--P3 position-held pseudo-target units. The selected learning
rate, encoder weight decay, and epoch count are frozen across P4 budgets 7, 35,
100, and 500. P4 query labels cannot enter fitting, selection, checkpointing, or
prediction generation.

The primary effect is paired held-condition Macro-F1 against the completed
frozen-linear parent at the identical fold, source seed, support seed, support
observations, and query observations. Practical reliable recovery requires a
point gain of at least +0.05 and a paired 95% interval whose lower endpoint is
greater than zero. The exact ordered classification rules and stopping rule are
in the normative JSON.

