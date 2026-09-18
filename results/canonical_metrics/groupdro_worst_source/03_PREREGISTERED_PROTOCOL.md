# Preregistered GroupDRO protocol

The primary source groups are P1, P2, and P3. Matched ERM and GroupDRO share the canonical C1 representation, source folds, seeds, optimizer, checkpoint rule, and balanced sampler. GroupDRO alone uses detached exponentiated-gradient q updates over source positions with the frozen eta grid 0.01, 0.05, 0.10, and 0.20. P4 labels remain sealed until predictions are written and hashed.
