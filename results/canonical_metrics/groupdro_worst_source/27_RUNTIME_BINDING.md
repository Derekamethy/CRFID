# Governed runtime binding

Status: `PASS_GOVERNED_RUNTIME_BINDING`.

- Source artifact root: external governed Strict-DG runtime, relative `source_inputs`.
- P4 root: external governed P4 CSV root containing hash-locked `A1_P4.csv`, `A2_P4.csv`, and `A3_P4.csv`.
- Source bundle fingerprint: `b4388de95eab0123178b57e29cfdbc69ac656666101038aca5cf2ca174b32164`.
- P4 bundle fingerprint: `60a65a6117b5625b8b510b21e110a5eac469632a485612c6f5277e8ffa423eef`.
- Canonical loader: `crfid.strict_runtime.neutral_data.CanonicalPhase2Data`.

The workflow reads P4 signals and opaque block identifiers before prediction freeze. TagID and ER are opened only after final predictions are frozen and hashed. Governed measurement data and runtime checkpoints remain outside the public repository.
