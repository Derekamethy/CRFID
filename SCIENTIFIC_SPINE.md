# Scientific spine

## 1. Familiar-condition learning is not unseen-domain transfer

The 12,600-row Tyndall/Paper4 dataset is fully factorial across TagID, ER,
position, surface, and repetition. Earlier random/grouped or within-position
evaluations show that TagID can be learned when acquisition conditions are
represented. Those evaluations answer a useful but narrower question than
transfer to an unseen reader position.

## 2. Strict-DG isolates the P4 shift

The governed Strict-DG protocol fits preprocessing, candidate models,
hyperparameters, architecture, seeds, and checkpoints using P1-P3 only. P4 is
held outside the development artifacts and is opened only after the source
decision is sealed. C1 (first-difference, GroupNorm 1-D CNN ERM) was selected
from source evidence and evaluated over seeds 42-46.

The frozen P4 result is near chance: accuracy `0.15663492063492063` and
Macro-F1 `0.11223088363457531`. Frozen predictions reproduce those aggregates
exactly. This is the canonical source-only result.

## 3. Tested DG objectives do not reliably improve P4

IRM, DANN v2, and GroupDRO each use a branch-local matched ERM comparator.
Their P4 Macro-F1 effects are respectively about `-0.01933`, `+0.01604`, and
`+0.00141`; every uncertainty interval crosses zero. Cross-branch absolute
scores are contextual because sampling, estimands, and matched controls differ.
Position suppression from DANN was not reliable, and the causal role of position encoding remains unresolved.

Domain-aware Mixup is not another negative performance row. Its frozen split
supports only 50% of required lawful cross-domain parents, so execution stopped
before development and P4 evaluation.

## 4. Limited target-labelled adaptation does not rescue this representation

Factor-aware few-shot, a larger 0-500-label calibration curve, a trainable
linear readout, and partial/full encoder fine-tuning remain at or near chance
under their governed query protocols. These are target-labelled diagnostics,
not source-only DG results.

The negative adaptation findings do not prove that P4 contains no TagID
information. The same embedding can achieve very high accuracy within
represented conditions while performing near chance across new conditions,
which is evidence of condition entanglement.

## 5. Historical target-informed performance is lineage-specific

A separate historical representation carrier reaches accuracy about `0.5771`
and Macro-F1 about `0.5836` after retrospective P4-informed readout
promotion. The same historical carrier already recorded about `0.5356`
Accuracy under its source-only selector. Because the carrier development
lineage differs from canonical C1, full-P4 outcomes influenced readout
promotion, and there is no independent target test, this result does **not**
establish that the canonical C1 representation contains substantial
recoverable P4 information.

The matched target-assisted study likewise does not support a general selector
gain. Its near-perfect within-condition A2 value is an interpolation result and
must not be described as unseen-condition generalisation.

## 6. Diagnostics point to a joint acquisition/representation limitation

At the encoder-clean P4 comparison, RAW Macro-F1 **0.2427** versus C1 **0.1472**, an observed mean contrast of **+0.0955**. The preregistered primary paired TagID-stratified block bootstrap gives an estimate of **+0.0951**, 95% CI **[0.0173, 0.1747]**. A block-plus-probe-seed sensitivity interval crosses zero, **[-0.0015, 0.1891]**. The primary evidence supports reduced linearly accessible TagID discrimination in C1 at P4, while seed-level uncertainty weakens the strength of that inference; it does not establish loss of all TagID information.

The synchronised physical analysis supports an adverse association with
45-degree acquisition in this campaign, but does not establish angle as the
unique or dominant cause. The overall distance main effect is not confirmed;
the negative 0-degree simple distance effect is supported for pooled
condition-block Macro-F1, while seed-inclusive Accuracy uncertainty crosses
zero. The interaction remains unconfirmed and floor-limited.

The pre-specified peak descriptor failed its global extraction-validity gate:
missingness is **lowest at P4**, not highest. Missing peaks therefore do not
explain the held-P4 transfer failure.

## 7. OpenEMS is an exploratory redesign bridge

OpenEMS connects the measured diagnostics to a simulation hypothesis. It does
not validate a new classifier or hardware design. The historical best simulated
change was `0.681%`, below the 5% confirmation gate; pilot confirmation also
showed substantial mismatch. The correct final classification is
`NOT_CONFIRMED`, and the robust R2 path remained blocked with zero new
simulations.

## Final bounded conclusion

The project supports a coherent negative result: strong familiar-condition
learning does not translate into reliable unseen-P4 generalisation for the
tested source-selected representation and methods. A separate historical
carrier can score substantially higher under retrospective target-informed
selection, but that result does not establish recoverability in canonical C1.
The diagnostics support a joint acquisition-dependent signal and
representation-transfer limitation, with possible residual readout mismatch.
The evidence does not support a universal impossibility claim, a unique causal
physical proof, or a validated OpenEMS redesign.
