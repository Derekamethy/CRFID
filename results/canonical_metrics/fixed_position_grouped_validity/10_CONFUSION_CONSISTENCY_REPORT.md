# Confusion consistency report

All 60 primary runs passed the compact confusion-matrix consistency checks. Each row-level matrix sums to 1,050, each condition-block matrix sums to 21, each predicted-class histogram sums to 1,050, and every histogram equals the corresponding confusion-matrix column sums.

The committed matrices contain counts only; logits and full predictions remain under the ignored runtime boundary and are not committed.
