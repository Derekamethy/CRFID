# Pre-DG Frozen Protocol

## Data and splits

The custody layer verifies the processed array, metadata, canonical manifest,
all 18 split CSV files, their metadata, row counts, class order, sample
identity, and raw-condition group disjointness before training.

Paper3 split counts are:

- grouped-random: 5,760 train / 1,920 validation / 1,920 test;
- each leave-position: 5,760 / 1,440 / 2,400;
- each leave-surface: 6,080 / 1,600 / 1,920.

Paper4 split counts are:

- grouped-random: 7,350 / 2,450 / 2,800;
- each leave-position: 7,350 / 2,100 / 3,150;
- each leave-surface-case: 6,650 / 1,750 / 4,200.

## Representation and preprocessing

The native physical signal is linearly interpolated in float64 to 512 points,
then stored as float32. Channel 0 is the measured signal and channel 1 is the
historical all-zero compatibility channel. No gradient channel is used.

For each split, channel mean and population standard deviation (`ddof=0`) are
fit on train only over sample and length axes. Exact zero standard deviation is
replaced by one. The float32 preprocessing state is serialized and reloaded
before transformation. Validation and test never fit preprocessing.

## Model

Both datasets use the frozen DeepCNN:

1. `Conv1d(2,128,7,padding=3)` + batch normalization + ReLU + max-pool;
2. `Conv1d(128,256,7,padding=3)` + batch normalization + ReLU + max-pool;
3. `Conv1d(256,256,7,padding=3)` + batch normalization + ReLU + max-pool;
4. `Conv1d(256,128,7,padding=3)` + batch normalization + ReLU + max-pool;
5. `Conv1d(128,64,7,padding=3)` + batch normalization + ReLU +
   adaptive average pool to length 8;
6. flatten, `Linear(512,512)`, ReLU, frozen dropout candidate, classifier.

The parameter count is 1,245,896 for Paper3 and 1,245,383 for Paper4.
PyTorch seeded default initialization is used.

## Training and selection

Training uses cross-entropy and AdamW (`lr=0.001`, weight decay `0.00001`),
batch size 512, at most 50 epochs, no scheduler, and patience 10 with minimum
improvement `1e-6`. Training is shuffled; validation and test are not.

The best epoch maximizes validation accuracy and then minimizes validation
loss. The candidate rule uses the same ordering, then lower dropout as the
final tie-break. Test data is not used for checkpoint or candidate selection.
All rules are executed for seeds 42, 43, and 44.
