# OpenEMS redesign evidence

OpenEMS was used as an exploratory bridge from measured CRFID failure diagnostics to geometry-redesign hypotheses. It was **not** a classifier result and did not validate a fabrication-ready tag.

Final classification: **NOT_CONFIRMED**.

The confirmed baseline worst-pair separation was about **3.1929 dB** and the selected candidate about **3.2146 dB**, a minimum-pair gain of approximately **0.681%**. This is below the preregistered **5%** practical-improvement gate. Pilot-to-confirm mismatch was also substantial, including a maximum reported peak shift of about **550 MHz**.

The robust R2 follow-up executed **zero new governed simulations** because the locked runtime could not import compatible openEMS/CSXCAD bindings; optimisation therefore remained blocked.

Files:
- `confirmed_redesign.json`: final assumption-bounded redesign assessment.
- `ranking_confirm.json`: confirmed candidate ranking and diagnostics.
- `r2_baseline_gate.json`: R2 stop-gate record.

No hardware classification validation was performed, and this evidence does not support fabrication recommendation.