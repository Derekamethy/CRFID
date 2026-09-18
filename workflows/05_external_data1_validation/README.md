# External Data1 Pipeline Validation

This workflow reproduces the frozen within-Data1 set-3 versus sets-4–9
pipeline validation. It trains local four-class models from scratch and does
not load source-dataset weights.

Set `CRFID_EXTERNAL_DATA1_ROOT` to the manifest-authorized nine-file raw root,
set `CRFID_EXTERNAL_DATA1_REFERENCE_ROOT` to the frozen result archive for
comparison, set `PYTHONHASHSEED=0` before interpreter startup, validate the
locked environment, then use the governed Python launcher with `run.py`.

EV4 is intentionally excluded from this runner and must be executed by the
independent verifier. EV4 has completed with verdict
`PASS_EV4_WITH_REQUIRED_CAVEATS`. This is within-Data1 pipeline validation, not
direct seven-class Paper4 external validation.
