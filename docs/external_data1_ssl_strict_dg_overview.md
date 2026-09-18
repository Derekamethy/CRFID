# External Data1 SSL under source-only DG

This study tests whether self-supervised pretraining with an external four-class Data1 corpus improves sealed Paper4 P4 performance without using P4 for model development.

Treatments share the same downstream seven-class evaluation setting. The matched supervised-only treatment T0 reached mean head Macro-F1 **0.12394**. The best SSL mean was lower (T4 about **0.10910**), and no tested SSL treatment produced a reliable or practically meaningful improvement over the matched supervised-only control.

For the main external-data contrast, joint Paper4-source + Data1 SSL versus Paper4-source-only SSL changed mean Macro-F1 by only about **+0.00376**; only 2 of 5 seeds improved.Data1 and Tyndall do not have an assumed class mapping. Data1 is used as auxiliary unlabelled signal data in this branch, not as a four-to-seven-class label-transfer source.

P4 remains sealed until final evaluation in the governed workflow. The result is therefore a scoped negative finding for the tested SSL treatments, not a general claim about self-supervised learning.

Compact results are in `results/canonical_metrics/external_data1_ssl/`; implementation is under `src/crfid/external_data1_ssl_strict_dg/` and `workflows/08_external_data1_ssl_strict_dg/`.