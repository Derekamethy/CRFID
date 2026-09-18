# Project overview

This repository studies CRFID TagID classification under reader-geometry shift. The central benchmark develops models on P1-P3 and evaluates the held P4 joint geometry under a strict source-only protocol.

The project separates four information regimes: familiar-condition baselines, source-only domain generalization, target-labelled diagnostics, and retrospective/supporting analyses. Those regimes answer different questions and their scores are not interchangeable.

The main source-only branches are canonical ERM, IRM, DANN v2, GroupDRO, and a domain-aware Mixup feasibility study. Supporting diagnostics examine representation transfer, fixed-position learnability, angle/distance effects, source-selection validity, limited P4 adaptation, external-data portability, and an exploratory OpenEMS redesign.

For the shortest reading path, use [Scientific spine](../SCIENTIFIC_SPINE.md), [Project status](../PROJECT_STATUS.md), and [Experiment registry](../EXPERIMENT_REGISTRY.md).
