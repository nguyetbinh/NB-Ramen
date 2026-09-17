# Implementation review — 2026-09-07

Independent read-only code reviewer: DONE, no concrete correctness blockers in
baseline parity, same-class swaps, one-shot labels, optimizer state/reset or
summary interpretation. The actual SignSGD constructor supplies only lr, leaving
momentum and decay at zero. Returned logits are cloned before diagnostic trials.
The reviewer noted that initial parity coverage used a small network/B4/k1/m2.
A subsequent real-autograd regression now exercises B100/k5/m10/capacity750,
16 ID queries and all 50 legal swaps/query on a small two-class network. This is
mechanics coverage, not CLIP effectiveness evidence.

The final eligibility implementation follows the source pseudocode: require at
least one class with m entries to qualify a query, then include available extras
from other classes with k < size < m. A regression checks that a partially filled
class is included without independently qualifying an otherwise ineligible query.

Full suite: 365 tests run, 360 passed, 5 skipped. Includes an actual MPS FP16
forward/backward probe test. Syntax compilation and git diff whitespace check pass.
The full output is in `unit-tests.log`.

An official-data attempt exposed PyTorch 2.4.1 MPS Half advanced-index backward:
`reference[mask].float()` fails during scatter backward. A minimal MPS reproduction
failed for that ordering and passed for `reference.float()[mask]`. The latter is
used for the evaluator-only supervised loss. Ramen cache weights/dtypes remain
unchanged.

Known limits: CPU/tiny-network assertions do not establish scientific headroom.
Pilot B reports only a screened lower bound, and its pooled correlation is
selection-biased. Undefined correlation/empty rates are null. No numerical GO
threshold was invented and no method-effectiveness conclusion is certified.
