# Few-Shot P4 Claims and Boundaries

## Claims this branch supports

1. With a frozen source-only encoder and head, both a target-only prototype
   adapter and a source-anchored linear head improve P4 macro-F1 over the
   zero-shot reference at 3 and 5 labelled samples per class, on paired episodes.
2. Overall accuracy stays close to the zero-shot level across all shot
   conditions. The macro-F1 gain reflects a redistribution across classes rather
   than a large accuracy jump.
3. At matched shots the two adapters rank differently: the head is ahead at
   1-shot, prototypes are ahead at 3-shot and 5-shot.
4. Head-recipe selection performed on P1-P3 pseudo-target evidence did not
   predict the P4 ranking. This is evidence that P4 is a materially different
   shift from held P1-P3 positions.
5. Class-level behaviour is heterogeneous: Tag-6 and Tag-7 improve under
   adaptation while Tag-5 does not.
6. Calibration cost is small in absolute terms: 7, 21 or 35 labelled samples
   against a 1,400-sample query set.

## Claims this branch does not support

- **Not strict domain generalization.** P4 labels are used. The strict
  source-only result is a separate branch and is neither reproduced nor
  superseded here.
- **No query-label use in adaptation.** Query labels were sealed before any
  numerical target access and opened only after every Stage-A prediction was
  written and hashed. They never entered adaptation, hyperparameter selection,
  early stopping, checkpoint selection, seed selection, episode design or
  preprocessing.
- **Not "no target labels were ever used".** Support labels were used, by design.
  Any statement that this branch avoided all P4 labels would be false.
- **No deployable combined adapter.** The two adapters were evaluated separately.
  The prototype 3-shot row is the best observed value among separately evaluated
  methods, a descriptive maximum only. No per-shot or per-class hybrid is
  claimed.
- **No causal mechanism.** The centroid-shift, boundary-shift and
  representation-overlap readings of the per-class patterns are consistent with
  the evidence but are not proven.
- **No universal superiority.** The bootstrap intervals are supporting
  descriptive uncertainty over 18 paired episodes.
- **Not the retrospective target-informed line.** The historical figure of
  roughly 0.577 accuracy is labelled
  `RETROSPECTIVE_TARGET_INFORMED_DEVELOPMENT_REFERENCE`, belongs to a different
  branch, and is not a like-for-like budget or protocol comparison.
- **Historical replay requires external artifacts.** The public repository retains compact results and verification code; full replay requires the omitted frozen archive. No fresh training or inference is claimed by the compact results.
