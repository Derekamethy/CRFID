# Data Format

`SampleMetadata` defines sample identity, dataset, class, domain, surface, condition and repeat. `SignalRecord` pairs metadata with a finite one-dimensional spectrum and optional monotonic frequency axis. Labels are validated against the declared task class order. Protected split groups are condition identifiers, not individual rows.

Tyndall uses seven local classes and positions P1–P4. Data1 uses four local classes and named measurement sets. No cross-dataset class equivalence is assumed.
