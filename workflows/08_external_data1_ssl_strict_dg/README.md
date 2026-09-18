# 08 — External unlabelled Data1 assistance under strict source-only DG

Does self-supervised pretraining with **unlabelled external Data1** signals improve Paper4
**P4** generalisation, when every selection, hyperparameter and training decision is made
without P4 labels, P4 metrics, P4 support samples or P4-informed historical choices?

The external-data contribution is measured as `T3 − T1`
(joint Paper4-source + Data1 SSL, minus Paper4-source-only SSL) — **not** as
`joint SSL − no SSL`, which would confound self-supervision with external data.

## Treatments

| ID | What it is |
|---|---|
| `T0_MATCHED_SUPERVISED_ONLY` | matched control: no SSL, no Data1 |
| `T1_PAPER4_SOURCE_ONLY_SSL` | unlabelled Paper4 P1–P3 SSL → supervised fine-tune |
| `T2_DATA1_ONLY_SSL` | unlabelled Data1 SSL → supervised fine-tune |
| `T3_JOINT_BALANCED_SSL` | 50/50 Paper4-source + Data1 SSL → supervised fine-tune |
| `T4_JOINT_PHYSICS_AWARE_SSL` | same corpus as T3, source-selected physics-aware objective |
| `T5_JOINT_DOMAIN_INVARIANT_SSL` | gated; adds gradient-reversal dataset-identity confusion |

## Stages

```
gate-a → screen → select → treatment (×5) → gate-t5 → finalize → evaluate-p4 → analyse → package
```

See `outputs/external_data1_ssl_strict_dg/21_REPRODUCTION_COMMANDS.md` for the exact
commands and the two required environment variables.

## Boundaries

- P4 is sealed until Gate D; `sealed_target.py` mints an access token only from a hashed
  preregistration and hash-verified checkpoints, releases labels only after predictions are
  closed, and refuses to release features again afterwards.
- Data1 labels are stripped at the file-reading boundary and never reach the model, loss,
  sampler or checkpoint selector.
- All writes are confined to `outputs/external_data1_ssl_strict_dg` by
  `paths.ensure_branch_output`.
- This branch is separate from `workflows/05_external_data1_validation` and
  `workflows/06_external_data1_portability` (within-Data1 pipeline validation) and from
  `workflows/05_few_shot_p4_adaptation` (which uses P4 support samples). It modifies none of
  them.
