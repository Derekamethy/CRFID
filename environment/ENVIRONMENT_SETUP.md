# Python environments

## Public development and tests

Python >= 3.11 is supported by `pyproject.toml`. Creating a local virtual environment and installing dependencies is supported:

```powershell
python -m venv .venv
./.venv/Scripts/python -m pip install --upgrade pip
./.venv/Scripts/python -m pip install -e ".[ml,test]"
./.venv/Scripts/python -m pytest -q
```

Use that interpreter consistently, or activate the environment first. Public tests and compact-evidence verification do not require a machine-specific environment identity. PyTorch is provided by the `ml` extra; historical plotting/HDF5 utilities use the `historical` extra. OpenEMS/CSXCAD are optional external solver dependencies.

## Historical governed execution

`CRFID_ENVIRONMENT_LOCK.json` and `requirements/crfid-strict-dg-lock.txt` record the exact CPU environment used for historical execution. `tools/assert_environment.py` checks that identity using an externally supplied machine configuration. These strict interpreter and configuration hashes apply only to an exact historical replay claim; they are not requirements for installing the public package or running its tests.

A supported public environment need not produce bitwise-identical retraining results. See [reproducibility](../REPRODUCIBILITY.md) for the distinction between compact verification, governed measurements, and omitted historical inputs.
