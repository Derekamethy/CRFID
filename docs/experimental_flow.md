# Experimental flow

The repository separates the project into scientific branches with different data-access rules. They should not be compared as if they were one sequential optimization pipeline.

```text
Canonical preprocessing and source modelling
|-- Pre-DG baseline
|-- Strict source-only DG on P1-P3 -> held P4
|   |-- ERM baseline
|   |-- IRM
|   |-- DANN v2
|   |-- GroupDRO
|   `-- domain-aware Mixup feasibility gate
|-- P4-labelled adaptation diagnostics
|   |-- few-shot calibration
|   |-- factor-aware support selection
|   |-- larger calibration curve
|   |-- trainable linear readout
|   `-- encoder fine-tuning
|-- Representation / physical diagnostics
|-- External Data1 portability and SSL
`-- OpenEMS exploratory redesign
```Strict-DG freezes model development before P4 access. Target-labelled branches are reported separately and cannot be used as evidence of unseen-target generalization.

External Data1 has a different class structure. Its labelled branch measures within-Data1 portability, while the SSL branch uses Data1 only as auxiliary unlabelled signal data for a sealed Paper4 transfer experiment.

OpenEMS is exploratory simulation evidence and does not establish a hardware classification improvement.