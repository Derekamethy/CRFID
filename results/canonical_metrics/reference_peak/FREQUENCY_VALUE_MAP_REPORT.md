# Tier 1A Frequency Value Map: Source-Only Attribution and Retrospective Transfer

## Technical summary

The frozen encoder assigns reproducible value to four physically verified spectral regions. Source-only leave-one-position-out (LOPO) development selected a **21-bin interpolated occlusion** because it had the highest cross-position rank stability (mean Spearman \(\rho=0.7480\)) among the six pre-registered configurations. The strongest stable score occurs in the L2 region at **6.6849 GHz**; the L1, L3, and common-resonator (Lc) regions also contain distinct peaks.

Three checks support the attribution shape: occlusion and integrated gradients agree strongly (\(\rho=0.8222\)); the learned-model map is essentially unrelated to the randomized-model map (\(\rho=0.0111\)); and ablating the five highest-ranked windows causes a larger mean accuracy loss than any of 20 random five-window draws (0.07563 versus 0.05728). These results support a diagnostic frequency ranking, not a causal or EM-validated redesign rule.

One frozen P4 pass gives moderate retrospective transfer (\(\rho=0.6454\), with 7 of the top 10 windows shared). Under the declared information-access protocol this is **`PROTOCOL_LOCKED_RETROSPECTIVE_TARGET`** evidence only. It must not be described as prospective fair-DG performance or used to retune the Level-A map.

## Stable source value concentrates in the verified resonator regions

The primary value measure is the mean decrease in cosine similarity to the true source-fitted class prototype after occluding a frequency window. For held source position \(p\), this is \(V_p(f)\). The stability-adjusted score is

\[
V_{\mathrm{stable}}(f)=\operatorname{mean}_{p\in\{P1,P2,P3\}}V_p(f)-
\operatorname{sd}_{p\in\{P1,P2,P3\}}V_p(f).
\]

Positive values mean that removing local spectral information reduces true-class similarity. The score is a conservative cross-position heuristic, not a confidence bound.

| Verified region | Conditional-frequency window (GHz) | Mean stable score | Maximum stable score | Frequency of maximum (GHz) |
|---|---:|---:|---:|---:|
| Lc | 5.100–5.745 | 0.00428 | 0.02107 | 5.2994 |
| L3 | 5.800–6.350 | 0.00757 | 0.02759 | 5.9452 |
| L2 | 6.450–6.950 | **0.01408** | **0.03496** | **6.6849** |
| L1 | 7.050–7.700 | 0.00914 | 0.03165 | 7.6419 |

The original analysis figure shows that the principal occlusion peaks fall inside the independently verified Lc/L3/L2/L1 regions. Integrated gradients recovers the same broad structure, while the permutation curve is used only as a shape comparator because it is on a different numerical scale. The compact public copy retains the numerical summary but not the historical figure bundle.

## Bit- and class-specific maps localize different failure channels

The per-bit result measures the change in bit-error rate after occlusion. Its largest positive peak is L1 at 7.6419 GHz (\(\Delta\mathrm{BER}=0.07514\)), followed by L3 at 6.0039 GHz (0.06082) and L2 at 6.7084 GHz (0.05062). Their mean changes within their own verified bands are 0.02499, 0.01982, and 0.02024, respectively. Thus the L2 band has the largest aggregate stable cosine damage, while the sharpest single bit-recovery failure occurs for L1; these are different outcomes and should not be collapsed into one ranking.

The class map provides a compatible decomposition. Tag 1 peaks in the L1 region (7.5303 GHz); Tags 4 and 5 peak near the L3 resonance (6.0039 GHz); and Tags 2, 3, 6, and 7 peak in the L2 region (6.5851–6.7671 GHz). Orange cells denote small improvements after occlusion and blue cells denote damage, so the map also exposes counterexamples to a uniformly positive “importance” interpretation.

The compact public copy retains the class- and bit-specific numerical interpretation without the historical figure bundle.

## Cross-position structure is stable; P4 agreement is only moderate

For the selected configuration, the source-fold rank correlations are 0.7341 (P1–P2), 0.7650 (P1–P3), and 0.7449 (P2–P3). This supports a common source-domain ordering despite meaningful amplitude differences between positions.

The frozen P4 comparison preserves much, but not all, of that ordering: the stable source curve and P4 damage curve have \(\rho=0.6454\), and 7 of their top 10 occlusion windows overlap. The unperturbed P4 accuracy is 0.5771428571, reproduced as an existing-artifact integrity anchor. The P4 curve is standardized only for visual rank/shape comparison; its amplitude is not treated as calibrated to the source curve.

The compact public copy retains the position-stability and retrospective-P4 statistics without the historical figure bundle.

## Scope, data, and metric definitions

- **Encoder and readout:** the archived `margin_refine_v3b` five-view encoder is frozen; class scores are cosine similarities to normalized nearest-class-mean prototypes.
- **Level-A cohort:** 7,350 balanced training rows from P1–P3 (1,050 rows per tag). The held-position fold sizes are P1=2,400, P2=2,450, and P3=2,500. Prototypes are fitted on the other two source positions.
- **Level-B cohort:** 3,150 P4 test rows, accessed once after the configuration was frozen and recorded in the target-access ledger.
- **Frequency axis:** 512 conditional bins from 5.0 to 8.0 GHz (approximately 5.87 MHz spacing). The selected 21-bin window spans about 117 MHz between its first and last bin; starts are separated by 10 bins, with an endpoint-aligned final window.
- **Primary outcome:** decrease in true-class prototype cosine similarity after occlusion. Secondary outcomes include margin change, accuracy change, prediction flips, and per-bit BER change.
- **Comparison basis:** rank correlation across all 512 bins. Curves expanded from overlapping windows contain ties and local plateaus.

This report summarises the saved 2026-07-17 Tier-1A analysis. The compact public repository retains the result report and lightweight reconstruction evidence rather than the complete historical CSV/JSON/figure bundle. The reported P4 comparison remains retrospective target-informed evidence and is not used as a Strict-DG benchmark.

## Experimental design and configuration choice

The script tested window widths 9, 15, and 21 with two replacements: linear interpolation across the missing interval and the prototype-fit-fold mean spectrum. Selection used the mean pairwise Spearman stability of the three held-position curves, with dynamic range only as a tie-breaker.

| Width (bins) | Replacement | Cross-fold stability | Mean-map dynamic range | Selected |
|---:|---|---:|---:|:---:|
| 9 | Interpolation | 0.3168 | 0.00605 |  |
| 9 | Fold mean | 0.5513 | 0.13082 |  |
| 15 | Interpolation | 0.6137 | 0.02440 |  |
| 15 | Fold mean | 0.5251 | 0.14655 |  |
| 21 | Interpolation | **0.7480** | 0.04779 | **Yes** |
| 21 | Fold mean | 0.6158 | 0.15588 |  |

Integrated gradients used 64 midpoint steps and a stratified sample of 70 rows per tag per fold (1,470 source rows in total). Its primary baseline is the corresponding prototype-fit-fold mean spectrum. Five seeded within-window permutations were evaluated. The model-randomization check rebuilt the same architecture from a fixed random seed, and the ablation check compared the top five selected windows with 20 random five-window draws.

## Robustness and lightweight consistency validation

| Check | Saved result | Criterion | Outcome |
|---|---:|---:|---|
| Learned versus randomized-model map | \(\rho=0.0111\) | \(|\rho|<0.3\) | Pass |
| Top-five mean accuracy loss | 0.07563 | Greater than random maximum | Pass |
| Random-five maximum accuracy loss | 0.05728 | Comparator | — |
| Median IG completeness gap | 0.00568 | \(\le 0.05\) | Pass |
| Occlusion versus IG agreement | \(\rho=0.8222\) | \(\ge 0.5\) | Pass |
| Deterministic occlusion flag | True | True | Pass |

A separate lightweight readback found 512 rows in each frequency CSV, contiguous bins 0–511, a 5.0–8.0 GHz endpoint match, and finite numeric values throughout. Average-rank Spearman recomputation gave 0.82265 for occlusion–IG and 0.64569 for stable-source–P4, consistent with the saved 0.8222 and 0.6454 after rounding and the script's index-rank tie convention. No expensive model rerun was performed.

## Limitations and uncertainty

1. **This is attribution of a frozen classifier, not an EM sensitivity model.** A high score identifies frequencies used by the encoder under the chosen perturbation; it does not establish that changing a dipole will cause the predicted performance change.
2. **Occlusion counterfactuals may be off-manifold.** Interpolation is less extreme than zeroing, but it can still produce spectra not generated by the physical system.
3. **LOPO measures map stability, not independent model generalization.** The encoder and frozen view statistics were trained/fitted on all source positions, including the position later held out for attribution assessment. The fold separation applies to prototype fitting and evaluation rows.
4. **`V_stable` is not inferential uncertainty.** Mean minus one cross-position standard deviation is a pre-registered ranking rule; it is not a confidence interval and does not account for sample-level dependence.
5. **The alternate IG baseline is mislabeled in the CSV.** `IG_zero_baseline` is produced from a constant raw spectrum equal to the sampled raw mean, not a literal all-zero raw spectrum. The primary fold-mean-baseline IG result and its agreement check remain correctly described.
6. **Rank correlations use a simple index-rank implementation.** Ties are not average-ranked in the production script. Independent average-rank recomputation changes the reported correlations by less than 0.001, so it does not alter the conclusions.
7. **P4 remains target-informed retrospective evidence.** The moderate transfer result cannot upgrade the source-only map to fair-DG evidence or justify further P4-tuned configuration choices.

Overall validation status: **ready to share as a technical diagnostic, with the above caveats**. It is not ready to serve as a simulation-backed geometry redesign specification.

## Recommended next steps

1. Use the source-only ordering to prioritize physically grounded investigation of the L2 region near 6.68–6.70 GHz, followed by the L1 region near 7.64 GHz and L3 near 6.00 GHz.
2. Carry both outcomes into later redesign work: stable true-class cosine damage for global ranking, and per-bit BER damage for reliability constraints. Do not substitute one for the other.
3. Before any geometry optimization, pass the Tier-2 EM validity gate with complete geometry, excitation, material-loss, and calibration specifications.
4. If Tier 1A is rerun for methods refinement, replace the custom Spearman helper with average ranks and rename the alternate IG baseline accurately. Such a rerun should remain source-only unless a separately governed retrospective assessment is explicitly authorized.

## Further questions

- Do the L2 and L1 peaks remain ordered under physically plausible local attenuation, shift, and broadening budgets from Tier 1C?
- Can a validated EM model reproduce the four measured attribution regions without using P4 outcomes for tuning?
- Which redundant code allocation best protects the L1 failure channel while preserving separability of all seven tag IDs?
