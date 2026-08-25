# Open-set post-hoc stability and cost reporting

## Scope

`src/evaluation/open_set_consensus_analysis.py` now exposes the stability and
cost evidence already checked by `validate_completed_run` for every completed
open-set method.  This is reporting only: it does not change the stream,
method, evaluator context, trace schema, or any thesis acceptance gate.

## Per-method evidence

Each method entry now includes:

- `worst_domain_id_accuracy` alongside the existing open-set metrics;
- `stability.negative_adaptation`, preserving either `computed` plus its rate
  or `reference_required` plus a null rate;
- `stability.post_shift_recovery`, preserving its computed shift records or
  its not-applicable state;
- `cost.synchronized_forward_latency` from the synchronized full forward
  interval, `cost.retained_memory`, and `cost.throughput`.

The analyzer rejects absent, malformed, or non-finite applicable summary
blocks.  It never replaces a missing measurement with zero or an estimate.

## Consensus overhead interpretation

For `ConsensusRamen` and `OracleConsensusRamen`, each complete paired cell now
contains `consensus_vs_ramen_cost_overhead`.  It reports differences and ratios
for total synchronized forward latency, throughput, and maximum retained
memory against the same-stream, same-seed Ramen cell.

The report labels these values `paired_total_path_proxy`: the evidence schema
does not isolate consensus computation without invasive synchronization that
would alter the timed path.  Thus a latency difference is usable as a paired
cost signal, but not as a claim that all of that difference is consensus-only
work.

## Verification

```text
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest \
  tests.test_open_set_consensus_analysis -v
```

Result: 6 tests passed, covering stability/cost extraction, the paired
ConsensusRamen overhead proxy, missing-cost rejection, evaluator-context
isolation, and stream-identity pairing.
