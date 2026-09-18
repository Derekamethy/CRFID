# DANN position intervention executive summary

Classification: `POSITION_SUPPRESSION_NOT_ACHIEVED`.

GRL validity passed: `True`. Selected nonzero lambda: `0.30`. Source position-probe balanced-accuracy change: `-0.005397` (95% paired interval `(-0.01904761904761905, 0.008253968253968253)`). Source held-position Macro-F1 change: `0.006064`.

P4 block Macro-F1 change: `0.011201` (block-only 95% interval `[-0.037457116323671, 0.0595299262701935]`; block-plus-seed interval `[-0.061179446140448714, 0.08452198981143186]`). P4 block Accuracy change: `0.028571` (block-only 95% interval `[-0.02222222222222222, 0.08253968253968254]`).

Position decodability decreased at the point estimate: `True`; confirmed below zero: `False`. Source TagID retention guardrail passed: `True`. DANN improved P4 block Macro-F1 at the point estimate: `True`; survived block uncertainty: `False`. DANN improved P4 block Accuracy at the point estimate: `True`; survived block uncertainty: `False`. A target gain survived both block and training-seed uncertainty: `False`. Confirmed position suppression and transfer improvement occurred together: `False`. Mechanistic evidence was strengthened by the preregistered criteria: `False`.

P4 predictions were frozen and hashed before labels were opened. The intervention is not formal causal mediation and cannot prove physical causality.
