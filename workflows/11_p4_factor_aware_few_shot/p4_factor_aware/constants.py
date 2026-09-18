"""Immutable preregistered constants for the isolated P4 patch."""

from __future__ import annotations

CLASS_ORDER = tuple(range(7))
ER_ORDER = tuple(range(3))
SURFACE_ORDER = tuple(range(3))
ALL_CELLS = tuple((er, surface) for er in ER_ORDER for surface in SURFACE_ORDER)

QUERY_FOLDS = {
    0: ((0, 0), (1, 1), (2, 2)),
    1: ((0, 1), (1, 2), (2, 0)),
    2: ((0, 2), (1, 0), (2, 1)),
}

CHECKPOINT_SEEDS = (42, 43, 44, 45, 46)
SUPPORT_SEEDS = (104729, 130363, 155921, 181081, 205439)
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_REPLICATES = 10_000

RANDOM_BLOCK = "RANDOM_BLOCK"
ER_BALANCED = "ER_BALANCED"
SURFACE_BALANCED = "SURFACE_BALANCED"
JOINT_FACTOR_COVERAGE = "JOINT_FACTOR_COVERAGE"

BUDGET_STRATEGIES = {
    1: (RANDOM_BLOCK,),
    3: (RANDOM_BLOCK, ER_BALANCED, SURFACE_BALANCED, JOINT_FACTOR_COVERAGE),
    5: (RANDOM_BLOCK, JOINT_FACTOR_COVERAGE),
}

FS0 = "FS0_FROZEN_SOURCE_HEAD"
PROTOTYPE = "TARGET_ONLY_COSINE_PROTOTYPE"
LINEAR_HEAD = "FROZEN_ENCODER_LINEAR_HEAD_ADAPTATION"
ADAPTATION_METHODS = (PROTOTYPE, LINEAR_HEAD)

HEAD_LAMBDAS = {1: 100.0, 3: 100.0, 5: 0.1}
HEAD_SUPPORT_FIT_ACCURACY_THRESHOLD = 0.95
HEAD_SUPPORT_FIT_MACRO_F1_THRESHOLD = 0.95

P4_FILE_HASHES = {
    "A1_P4.csv": "8f63f8bee3deb7747a88ac0fe1f47eabe8c0feedfe5a65cb503e25fdcf1377ee",
    "A2_P4.csv": "6b20fb815d6df40ff9ad4c0ecbf3bc7a82762ac96b8fd7df7170504715d75a00",
    "A3_P4.csv": "6c3bffcca0e81664398423c5c4828c3d7d4698a69e804fe69368e53bb8a9f6e3",
}

CHECKPOINT_FILE_HASHES = {
    42: "cc507dcf895b2197d4a98c7c2eeb4c6520911944464da637b5002e1b7f8f9437",
    43: "3db0c7d3c3678d0d0b4f78714a86b49759e6526bbbac89a1bb78cd9c1266a6a3",
    44: "4acc015a59aed08faab2492c0cd2639d5c7de711c19c3bf1dcbb9cdda85453ce",
    45: "8a091de402a9750f5bd0443741afbf694b0155d2c008e6c0aa68123b62b71896",
    46: "1cdb5c787ba674e3b5a73fdf03ba0d378f5b46055792408ac4b4ea33a389c879",
}

CHECKPOINT_STATE_HASHES = {
    42: "5b5b8d778d1467c8d751676b316d9ac9e6c1605f634c4e85cde1d827ec8275d3",
    43: "5f3f0360e559a6522a1af03e57b98c917096f3889afac5a0bfa65b70d9fa1210",
    44: "a39e7a2c9ddc9593d812322d95b18129810cee24ba5a66c24f014d3bb4a53e96",
    45: "468c6f1a94ee4ddc16678bc2cc572eccae7f2485220ace919004504da336c776",
    46: "a3734a84be6d03e4376772aba8b32ae2d7a46d95f731cc326f616aa94619700a",
}

PREPROCESSING_FILES = {
    "FINAL_SOURCE_PREPROCESSING_MEAN_FLOAT64.npy": "a8054ebeb3ab6559d49220e530f7ef03996bd1598b070a3907cf390658baca52",
    "FINAL_SOURCE_PREPROCESSING_SCALE_FLOAT64.npy": "8709fd10263b60d3384e6b5b198e27314d8aefa85eb49b9c4c3ae0ccd1e9cecb",
}

HISTORICAL_PAYLOAD_MANIFEST_SHA256 = (
    "d5645c8840504a60d3126ecea585ceeff9f3bd1909fd297fd899c4f51e68d249"
)

HISTORICAL_BINDING_HASHES = {
    "src/few_shot/frozen_encoder.py": "761b601ccff3205b822e49d3eea32dafce6f3ad584093e686284df899f45a3fc",
    "src/few_shot/prototypes.py": "72762cfd1eb78332b5abedd90a20b2e92c1804f39e51b20de2fe5ceca935e742",
    "src/few_shot/source_head_adaptation.py": "953f7564a0f5b8deb003d2d5ccf354682ad49edc41a2818fbb193467f7835c3a",
    "04_prototype_adaptation/PROTOTYPE_METHOD_FREEZE.json": "30c18a351b65cbbd3d11c85bcbaf2935bdb8b57e432b1e25563d45eed5cc81aa",
    "05_head_finetuning/fs4_source_recipe_selection/HEAD_METHOD_FREEZE.json": "4f57b148f6dbbebe1b76bb9def042fc35941a34cb2a01c26c9e22f457e58bfd1",
    "02_support_query_protocol/EPISODE_MASTER_MANIFEST.json": "4aaff45e78d91b7bf9ea4014ec276cf2f08d230d9684b1e46ca423e1c7f3ca0b",
    "02_support_query_protocol/SUPPORT_SAMPLE_MANIFEST.csv": "72e1d1ef945ac1a1734bb7410196174303d5c27a93ca2b700d03a5f750ad31e0",
    "02_support_query_protocol/sealed_query_labels/QUERY_CLASS_MAPPING.json": "59340dba9ecfc4c91e4168c7b682a745a1dd155f48f0010beb5758f9f9c5033a",
    "02_support_query_protocol/P4_STRUCTURE_AUDIT.json": "229f3b1e2ff312840987053ba9746dc390f66984a30965fbc8e86ed4f37f761b",
    "01_frozen_source_import/frozen_last_code_snapshot/10_final_recipe_freeze/FINAL_SOURCE_PREPROCESSING_STATE.json": "18bae13a99f98f9284706b36c9e91c458cf19abdc611645fd8c78365a14630d2",
    "01_frozen_source_import/frozen_last_code_snapshot/11_final_p4_evaluation/FINAL_CHECKPOINT_MANIFEST.json": "46c34ffc51eeca942135393aa98c78cca295c23fe4be1e110c8549809d5c7558",
    "06_final_comparative_audit_and_archive/FINAL_CANONICAL_RESULTS_TABLE.csv": "81165de9ada22c767a3a6de16d931a864c0f60ece4715819f4386aa2792d5481",
    "06_final_comparative_audit_and_archive/FINAL_METHOD_COMPARISON.csv": "6e63489fa572ed9ca91d2c324fc4ef8197b1524f8dc736adc2cdd1ef72f8a295",
    "04_prototype_adaptation/fs3_execution/FS3_UNIT_RESULTS.csv": "19f54cb4807e441cc2d70ee1774c20c083d08ff9968f9f646ec8c959db4395d6",
    "05_head_finetuning/fs5_p4_execution/FS5_UNIT_RESULTS.csv": "e3ba57bcb47a335e278874e45bc31829338f5defbff2b61eccbf8cad56a53456",
}

EXPECTED_ROWS = 3150
EXPECTED_BLOCKS = 63
ROWS_PER_BLOCK = 50
RAW_SIGNAL_LENGTH = 281
DIFFERENCED_SIGNAL_LENGTH = 280
EMBEDDING_DIMENSION = 256


def cell_id(cell: tuple[int, int]) -> str:
    """Return a publication-safe factor-cell identifier."""

    return f"E{cell[0]}xS{cell[1]}"
