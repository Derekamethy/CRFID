# Condition-matched cross-position manifold mixup preregistration

This source-only benchmark compares matched C1 ERM, within-position same-condition manifold mixup at alpha 0.50, and cross-position condition-matched manifold mixup at the frozen alpha grid `[0.2, 0.5, 1.0]`. Every mixed pair matches TagID, ER, and surface; cross-position pairs differ in position, while control pairs share position and use a different repeat where feasible.

Mixup is applied only at the 256-dimensional penultimate embedding during training. With `m ~ Beta(alpha, alpha)`, `m_effective = max(m, 1-m)`, `z_mix = m_effective*z_a + (1-m_effective)*z_b`, and beta fixed at 1.0, the objective is `(L_original + L_mix)/2`. Labels are not interpolated because pair labels are identical. Mixed embeddings are representation augmentations, not physical intermediate spectra.

Alpha selection is the specified source-only lexicographic rule. Final P4 predictions from five checkpoints for each of the three methods must be frozen, checkpoint-bound, hashed, and atomically receipted before P4 TagID or ER is opened. Primary P4 inference uses 63 TagID x ER x surface blocks, not 3,150 rows, with 10,000 paired TagID-stratified bootstrap replicates.
