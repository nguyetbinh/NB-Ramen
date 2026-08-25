# ConsensusRamen current-self retrieval ablation

`ConsensusRamen` now declares `include_current: true` explicitly, preserving
the original Ramen-compatible batch-atomic cache visibility used by v0.

`ConsensusRamenNoSelf.yaml` is the preregistered causal-history ablation:
`include_current: false` retrieves and aggregates only supports retained before
the current forward, then admits the whole batch after `set_by_sample_grad`.
The temporary SignSGD update and post-inference reset are unchanged.

The CPU mechanics test uses a historical support at feature 10 with gradient 7
and a current support at feature 0 with gradient 11 for a query at feature 0.
With `topk=1`, the no-self path produces 7 and only then stores 11; the v0
path produces 11.  This establishes both ordering paths arithmetically without
using labels or evaluator data.
