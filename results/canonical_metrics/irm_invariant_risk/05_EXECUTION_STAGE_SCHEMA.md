# Strict execution-stage schema

- `SOURCE_LOPO_DEVELOPMENT`: fold S1/S2/S3, two training positions, one held source position, `target_position = null`.
- `FINAL_SOURCE_TRAINING`: no development fold or held position, `training_positions = P1_P2_P3`, `target_position = null`.
- `FINAL_P4_EVALUATION`: no development fold or held position, `training_positions = P1_P2_P3`, `target_position = P4`.

The final P4 record is rejected by the source-fold validator. Any ambiguity raises `FAIL_EXECUTION_STAGE_SCHEMA_DEFECT`.
