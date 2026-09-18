# IRMv1 results

## Main result

IRMv1 used the branch-local matched ERM control, 75 source-domain development runs, 10 final runs, and selected `lambda = 1` with a fixed annealing boundary at optimizer step 212.

| Metric | Matched ERM | IRM | IRM minus ERM | 95% interval | Interpretation |
|---|---:|---:|---:|---|---|
| Mean held-source Macro-F1 | 0.149769 | 0.163672 | +0.013903 | [-0.014233, 0.050652] | Directionally positive; not reliable |
| Worst held-source-position Macro-F1 | 0.121876 | 0.161286 | +0.039410 | [-0.022333, 0.060727] | Directionally positive; not reliable |
| P4 condition-block Macro-F1 | 0.118709 | 0.099379 | -0.019330 | [-0.066939, 0.025387] | No reliable benefit |
| P4 condition-block Accuracy | 0.133333 | 0.158730 | +0.025397 | [-0.025397, 0.076190] | Not reliable |

The P4 intervals above include both physical-condition-block and training-seed uncertainty. The Macro-F1 interval crosses zero, so the experiment does not show a reliable IRM benefit over its matched ERM control.

## Invariance diagnostics

| Diagnostic | Matched ERM | IRM | Difference | 95% interval | Finding |
|---|---:|---:|---:|---|---|
| Development outer-refit IRM penalty | 0.231437 | 0.192252 | -0.039185 | [-0.108348, 0.030150] | Lower numerically; not reliable |
| Environment-risk population SD | 0.015486 | 0.029972 | +0.014487 | [0.005188, 0.022851] | Dispersion increased |
| Position-probe balanced Accuracy | 0.901587 | 0.876190 | -0.025397 | [-0.092063, 0.012698] | Position remained strongly decodable |
| TagID-probe Macro-F1 | 1.000000 | 0.801342 | -0.198658 | [-0.445498, -0.053974] | Task information decreased |

Three of the 15 selected-lambda development outer-refit units received no penalised optimization steps. Restricting the sensitivity analysis to the 12 genuinely intervened units does not change the main conclusion.

## Boundaries

Lambda was selected using the same held-source endpoint later summarized as the source contrast, so the source point estimate is not an independent post-selection effect. P4 inference uses 63 condition blocks and five training seeds. Absolute scores from this branch should not be ranked directly against canonical ERM; the supported comparison is the branch-local matched effect.

P4 predictions were frozen and checkpoint-bound before labels were opened. The evidence does not establish successful invariance, a reliable P4 benefit, a reliable P4 degradation, or a causal physical mechanism.
