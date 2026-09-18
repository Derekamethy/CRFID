# Model and control design

All arms use the canonical `C1_FIRST_DIFFERENCE_ERM_1DCNN` encoder and TagID head, source-only preprocessing, AdamW settings, batch size 256, maximum 50 epochs, validation Macro-F1 checkpoint metric, and deterministic balanced position-by-TagID ordering.

The A1/A2 domain head is `Linear(256,128) -> ReLU -> Dropout(0.0) -> Linear(128,K)`. The domain head always minimizes ordinary cross-entropy. Only the encoder-side coefficient differs: A1 `+lambda(p)`, A2 `-lambda(p)`. A0 is canonical TagID ERM with coefficient zero.

Every fold-by-seed-by-lambda matched set uses identical sample identities, ordering, batch boundaries, class/domain exposure, initialization of shared parameters, optimizer, and epoch/checkpoint rules.

