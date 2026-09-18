# Source-domain peak separability

## Scope and denominator

This is a source-only supporting analysis. It uses the observed P1/P2/P3 per-tag peak
means and within-tag standard deviations from
`06_failure_mechanism/composition_warp/03_peak_warp_prediction/results/predicted_p4_peaks.json`
(fields `observed_per_tag_peak_ghz` and `observed_within_tag_sd_ghz`).
It does not use raw P4 spectra and is not a Strict-DG result.

The denominator is:

- 21 unordered tag pairs;
- 4 peaks (`L1`, `L2`, `Lc`, `L3`);
- 3 source positions (`P1`, `P2`, `P3`);
- **21 × 4 × 3 = 252 comparisons**.

## Reliable-separation definition and preregistration

The definition was fixed before the run in
`06_failure_mechanism/composition_warp/preregistration.md`, under
“§7 — predicted-collision threshold.” A peak comparison is counted as
reliably separated when the absolute difference between the two tag means is
strictly greater than **1.0 × pooled within-tag SD**.

Each tag has nine source-condition observations, so the equal-count pooled SD
used by the source result is:

`pooled_SD = sqrt((SD_tag_i^2 + SD_tag_j^2) / 2)`.

Applying that preregistered rule gives **13/252 = 5.1587%**, reported as
**5.16%**.

## Full list of 13 reliable comparisons at 1.0 × pooled SD

| # | Tag pair | Peak | Position | Separation (GHz) | Pooled SD (GHz) | Separation / pooled SD |
|---:|:---:|:---:|:---:|---:|---:|---:|
| 1 | 1–3 | L1 | P1 | 0.246429 | 0.177474 | 1.388531 |
| 2 | 2–5 | L1 | P1 | 0.258333 | 0.199241 | 1.296586 |
| 3 | 2–6 | L2 | P1 | 0.170238 | 0.155162 | 1.097163 |
| 4 | 3–4 | L1 | P1 | 0.247619 | 0.205365 | 1.205752 |
| 5 | 3–5 | L1 | P1 | 0.308333 | 0.153986 | 2.002345 |
| 6 | 5–6 | L1 | P1 | 0.201190 | 0.188243 | 1.068783 |
| 7 | 5–7 | L1 | P1 | 0.196429 | 0.165388 | 1.187681 |
| 8 | 1–7 | L1 | P2 | 0.213095 | 0.168099 | 1.267676 |
| 9 | 5–7 | L1 | P2 | 0.251190 | 0.189117 | 1.328227 |
| 10 | 1–4 | L1 | P3 | 0.184524 | 0.164164 | 1.124019 |
| 11 | 1–6 | L1 | P3 | 0.220238 | 0.166508 | 1.322690 |
| 12 | 4–5 | L1 | P3 | 0.196429 | 0.164160 | 1.196571 |
| 13 | 5–6 | L1 | P3 | 0.232143 | 0.166503 | 1.394227 |

## Position and peak distribution

By position, the 13 comparisons are distributed as **P1: 7**, **P2: 2**,
**P3: 4**. P1 contributes the largest share, but the finding is not confined
to one position.

By peak, the distribution is **L1: 12**, **L2: 1**, **Lc: 0**, **L3: 0**.
The sparse reliable separation is therefore overwhelmingly an L1 effect.

## Threshold sensitivity

| Threshold | Reliable comparisons | Fraction | Percentage | Position distribution | Peak distribution |
|---:|---:|---:|---:|:---|:---|
| 1.00 × pooled SD | 13/252 | 0.051587 | 5.16% | P1: 7, P2: 2, P3: 4 | L1: 12, L2: 1 |
| 1.25 × pooled SD | 7/252 | 0.027778 | 2.78% | P1: 3, P2: 2, P3: 2 | L1: 7 |
| 1.50 × pooled SD | 1/252 | 0.003968 | 0.40% | P1: 1 | L1: 1 |
| 2.00 × pooled SD | 1/252 | 0.003968 | 0.40% | P1: 1 | L1: 1 |

The sole comparison retained at 2.00 × is tag pair 3–5, L1, P1, with ratio
**2.002345**. Because the criterion is strict (`>`), it remains above the
2.00 × threshold.

## Conclusion

The source-domain evidence supports low peak separability: the reliable share
is already only 5.16% at the preregistered threshold and decreases
monotonically under stricter multipliers. This conclusion is insensitive to
the tested threshold range, but it remains exploratory and source-domain only.

Separately, the §7 collision rule predicted **21/21 tag pairs** and is
non-discriminative: it produced a **100% complement false-positive/false-alarm
rate** in both few-shot (18/18) and strict zero-recall (15/15) evaluation.
