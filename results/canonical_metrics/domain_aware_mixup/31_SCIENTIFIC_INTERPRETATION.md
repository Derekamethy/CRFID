# Scientific interpretation

No scientific result is reported. The preregistered governed pairing gate stopped execution at 50.0% inner-training coverage.

The raw dataset contains complete TagID × ER × surface factorial support at every position. In the frozen canonical inner split, each source position retains 42 condition keys in `inner_train`, but only 21 are shared across the two source positions. The matched blocks for the other 21 keys exist in the raw dataset but were assigned to `inner_validation`, leaving 21 × 2 × 50 = 2,100 eligible parent samples out of 42 × 2 × 50 = 4,200. The failure therefore reflects `SPLIT_INDUCED_CROSS_POSITION_PAIRING_INCOMPATIBILITY` between the canonical inner split and the strictly matched cross-position augmentation rule, rather than incomplete raw-data support.
