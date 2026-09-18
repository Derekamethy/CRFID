# Reproduction commands

Historical scientific execution (do not rerun after release):

```powershell
crfid-python.cmd workflows/19_dann_v2/run.py bind --source-inputs <GOVERNED_SOURCE_INPUTS>
crfid-python.cmd workflows/19_dann_v2/run.py development --source-inputs <GOVERNED_SOURCE_INPUTS>
crfid-python.cmd workflows/19_dann_v2/run.py select
crfid-python.cmd workflows/19_dann_v2/run.py final-source --source-inputs <GOVERNED_SOURCE_INPUTS>  # only after a passing source gate
crfid-python.cmd workflows/19_dann_v2/run.py seal                               # only after a passing source gate
crfid-python.cmd workflows/19_dann_v2/run.py freeze-p4 --p4-directory <GOVERNED_P4_DIRECTORY>
crfid-python.cmd workflows/19_dann_v2/run.py score-p4 --p4-directory <GOVERNED_P4_DIRECTORY>
crfid-python.cmd workflows/19_dann_v2/run.py report
```

Focused non-training validation:

```powershell
crfid-pytest.cmd tests/test_dann_v2_model.py tests/test_dann_v2_protocol.py tests/test_dann_v2_evaluation.py -q
```
