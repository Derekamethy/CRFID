# Limitations

- One P4 corpus supplies seven TagIDs, nine ER-by-surface cells, 63 condition blocks, and no independent acquisition campaign.
- Cross-validation makes every block query exactly once, but a block can be support in other outer folds; folds are not independent datasets.
- Fifty rows per block are repeated acquisitions. They enter adaptation as labelled measurements but are never inferential bootstrap units.
- Exact duplicate signals are confined within blocks. Random row sampling can therefore spend labels on duplicate acquisitions; unique-signal coverage is reported.
- Exact 10- and 20-label budgets cannot be perfectly class-balanced across seven TagIDs; counts differ by at most one and the remainder rotates across seeds/folds.
- Independent support coverage saturates at 42 blocks. Below saturation, label count and coverage co-vary, so their effects are not causally separable.
- The historical retrospective 0.5836 and matched within-condition 0.9835 Macro-F1 values are not matched endpoints; no valid percentage of either gap is claimed.
- Results are conditional on one frozen C1 representation, one cosine-prototype adaptation rule, five source seeds, and twenty support selections.
- Bootstrap intervals describe this crossed block/seed design; they do not create new tags, devices, environments, or physical campaigns.
- No encoder fine-tuning, transductive query adaptation, threshold tuning, or alternate-method search is included.
