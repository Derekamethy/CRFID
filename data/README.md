# External data contract

Raw measurements and processed sample arrays are **not distributed** in this
repository. Their publication/redistribution rights have not been confirmed.
Do not commit them unless the rights holder explicitly approves release.

Expected external layout:

```text
CRFID_DATA_ROOT/
  tyndall/
    A1_P1.csv ... A3_P4.csv
  data1/
    set_1.csv ... set_9.csv
```

Tyndall/Paper4 has 12,600 rows: 7 TagIDs x 3 ER conditions x 4 positions x
3 surfaces x 50 repetitions. Each row contains governed metadata followed by
an ordered 281-value spectrum. Raw labels 1-7 map explicitly to model classes
0-6. P1-P3 are the 9,450-row Strict-DG source pool and P4 is the 3,150-row
target pool.

Data1 files contain 1,601 ordered signal values and one local integer label
from 0-3. There is no assumed class mapping between Data1 and Tyndall.

Machine-readable schemas, counts, class orders, and available fingerprints are
in `schemas/datasets.json`. Configure external paths with environment variables
described in `../REPRODUCIBILITY.md`; generated manifests should store relative
source names and hashes, not measurements or machine-local absolute paths.
