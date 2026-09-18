# Few-shot P4 adaptation

This target-labelled study evaluates small P4 support sets under sealed support/query episodes. It is an adaptation experiment, not source-only domain generalization.

Historical episode-level prototype Macro-F1 values were approximately:
- zero-shot reference: **0.1073**
- 1-shot: **0.1122**
- 3-shot: **0.1451**
- 5-shot: **0.1444**

The paired episode analysis supports small gains at 3 and 5 shots, but all episodes reuse one P4 corpus. These intervals therefore describe within-corpus episode uncertainty, not independent acquisition campaigns.

See `RESULTS.md` for the retained method table.