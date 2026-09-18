# CRFID Domain Generalization under Reader-Geometry Shift

This repository contains the code, governed experiments, compact result evidence, and reproducibility material for a chipless-RFID (CRFID) sensing study conducted at Tyndall National Institute.

The central question is whether a TagID classifier that works under represented reader geometries can generalise to a **held joint geometry**, and—when it does not—where the failure occurs across the measured signal, learned representation, and classifier readout.

## Experimental design

The primary Tyndall/Paper4 corpus contains **12,600 measurements** from a complete factorial design:

| Factor | Levels |
|---|---:|
| TagID | 7 |
| ER/material condition | 3 |
| Reader position | 4 (P1-P4) |
| Surface | 3 |
| Repeated sweeps | 50 |
| Physical condition blocks | 252 |

Each position contains 63 physical condition blocks and 3,150 sweeps. P1-P3 form the source domains; P4 is the held 150-mm/45-degree joint geometry in the Strict-DG experiment. Repeated sweeps from the same physical condition are kept together rather than treated as independent deployment conditions.

## Evaluation regimes

The project deliberately separates several questions that are easy to conflate:

- **Familiar-condition / Pre-DG:** can TagID be learned when related acquisition conditions are represented?
- **Strict source-only DG:** can a recipe developed using P1-P3 transfer to P4 without target-assisted selection?
- **Target-labelled diagnostics:** how much can limited labelled P4 support, readout replacement, or fine-tuning recover?
- **Retrospective / within-condition analyses:** what do target-informed or dependence-permissive scores reveal about the failure mechanism?

Scores from these regimes are not interchangeable.
## Main results

The familiar-condition baseline reached **0.7344 Accuracy**, but the source-selected C1 first-difference 1-D CNN reached only **0.1566 Accuracy** and **0.1122 Macro-F1** on held P4 over five seeds.

Standard source-only DG interventions did not provide a reliable matched improvement:

| Method | P4 Macro-F1 effect vs matched ERM | 95% interval |
|---|---:|---|
| IRM | -0.01933 | [-0.06694, 0.02539] |
| DANN v2 | +0.01604 | [-0.03558, 0.06862] |
| GroupDRO | +0.00141 | [-0.04298, 0.03949] |

The comparison intervals include condition-block and training-seed uncertainty.

Domain-aware Mixup stopped before training because the frozen split retained only **50%** of the lawful cross-position parents required by its preregistered feasibility gate.

At P4, the encoder-clean linear-probe comparison gave RAW Macro-F1 **0.2427** versus C1 **0.1472**, an observed mean contrast of **+0.0955**. The preregistered primary paired TagID-stratified block bootstrap gives an estimate of **+0.0951**, 95% CI **[0.0173, 0.1747]**. A block-plus-probe-seed sensitivity interval crosses zero, **[-0.0015, 0.1891]**. The primary evidence supports reduced linearly accessible TagID discrimination in C1 at P4, while seed-level uncertainty weakens the strength of that inference; it does not establish loss of all TagID information.

The physical analysis supports an adverse association with **45-degree acquisition** in this campaign, while the overall distance main effect and angle-by-distance interaction are not confirmed. P4 is not uniquely unreadable under matched within-position evaluation; it is scientifically important because it is the **unobserved joint geometry**.

Peak missingness does not explain the transfer collapse. The four-peak descriptor failed its global extraction-validity gate, and missingness was lowest at P4.

## Target access and recovery boundary

Limited P4 labels produced small, unstable, or protocol-sensitive recovery. At 500 labelled P4 samples, a target-trained linear readout reached Macro-F1 **0.1427**; partial and full encoder fine-tuning reached **0.1319** and **0.1296**.

A separate historical carrier reached Accuracy about **0.5771** and Macro-F1 about **0.5836** after retrospective P4-informed readout promotion. That carrier has a different development lineage from canonical C1 and the same P4 corpus influenced selection and scoring, so this is not an independent Strict-DG result and does not establish canonical-C1 recoverability.

A permissive within-condition target-fitted linear rule reached Macro-F1 about **0.9835**. Once physical-condition dependence is removed, performance collapses toward chance; the high score is therefore an interpolation result, not evidence of unseen-condition transfer.
## OpenEMS redesign study

OpenEMS was used as an exploratory bridge from the measured diagnostics to tag-geometry hypotheses. The best confirmed simulated minimum-separation change was about **0.681%**, below the registered **5%** practical-improvement gate. The redesign is therefore **NOT_CONFIRMED** and no hardware classification improvement is claimed.

## Repository structure

- `src/crfid/` — reusable preprocessing, models, evaluation, DG, adaptation, diagnostics, and OpenEMS modules
- `workflows/` — experiment entry points
- `configs/` — experiment and access-policy configurations
- `results/canonical_metrics/` — compact result evidence grouped by scientific question
- `06_failure_mechanism/` — retained composition-warp mechanism diagnostic
- `tests/` — algorithm, protocol-boundary, and synthetic regression tests
- `data/schemas/` — dataset schemas and external-data contracts
- `docs/` — method/protocol documentation needed to understand the public experiments

Start with:
- [Scientific spine](SCIENTIFIC_SPINE.md) for the argument and claim boundary
- [Project status](PROJECT_STATUS.md) for experiment-level conclusions
- [Canonical result index](results/canonical_metrics/README.md) for the evidence packages
- [Workflow index](workflows/README.md) for executable experiment entry points
- [Documentation index](docs/README.md) for protocol and result notes
- [Experiment registry](EXPERIMENT_REGISTRY.md) for information-access differences
- [Reproducibility](REPRODUCIBILITY.md) for setup and external-input boundaries

## Reproducibility

Raw measurement data are not distributed in this repository. Public code, frozen configurations, compact result tables, five final Strict-DG checkpoints, preprocessing state, and selected confusion matrices are retained. Reproduction scope varies: the public Strict-DG command verifies retained evidence; the factorial diagnostic accepts governed measurements; other historical paths also require omitted frozen inputs or archives. See [reproducibility boundaries](REPRODUCIBILITY.md).

A lightweight setup is:

```powershell
python -m venv .venv
./.venv/Scripts/python -m pip install --upgrade pip
./.venv/Scripts/python -m pip install -e ".[ml,test]"
./.venv/Scripts/python tools/validate_configs.py
./.venv/Scripts/python tools/check_import_safety.py
./.venv/Scripts/python -m pytest -q
```

Some tests require external measurements or omitted historical artifacts and skip when those inputs are unavailable. The public Strict-DG verifier runs without either:

```powershell
./.venv/Scripts/python workflows/02_strict_source_only_dg/run.py --config configs/strict_dg/canonical.yaml --execute
```

## Scope of the conclusion

The evidence supports a **joint acquisition-dependent signal and representation-transfer limitation**, with possible residual readout mismatch, for the tested seven-class campaign and held P4 geometry. It does not establish that domain generalization is impossible in CRFID, identify one unique electromagnetic cause, prove that every decoder must fail, or validate a fabrication-ready redesign.
