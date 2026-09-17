# Oracle support utility — execution status, 2026-09-07

> Historical bounded attempt. Pilot A has since completed all 16 eligible queries
> and 1,760 swaps; see [completed Pilot A report](pilot-a-completed.md). Pilot B remains pending.


Implementation and local verification are complete. The requested scientific
pilots are **not complete**. No GO/NO-GO decision or measured headroom is available.

## Delivered

- Diagnostic-only `OracleSupportUtilityProbe`, preserving Ramen outputs, cache
  admission, weights, active-class averaging and SignSGD.
- One-use evaluator known-label/OOD/domain hand-off; ID-only supervised scoring.
- Exhaustive A and screened B configurations, same-class swaps, random/feature/
  cosine controls, first-order ranking and exact temporary updates.
- Per-query JSONL and summary sidecars, incremental/atomic output, source/config
  hashes, provenance validation, and a sequential fail-closed pilot runner.
- Separate runtime documentation; the canonical 252-run matrix is unchanged.

## Validation

365 tests ran: 360 passed, 5 skipped. This includes real autograd/SignSGD tests at
B100/k5/m10/capacity750 over 16 ID queries, baseline output equality, full swap
coverage, context validation, eviction provenance, reset on failure, partial
class eligibility, MPS Half backward and ineligible-query cost avoidance.
These small-network results validate mechanics, not CLIP research headroom.
See [implementation review](implementation-review.md) and [unit log](unit-tests.log).

## Real-data execution

The final bounded attempt selected official CIFAR-100-C and CLIP ViT-B/16,
MPS, split v1, OOD=.5, block=64, batch=100, seed=0, k=5/m=10, capacity750,
source budget400/domain and a 600-sample prefix. Official model SHA-256 and
CIFAR acquisition/inventory passed the repository's fast provenance checks.
The available host is Apple arm64 with 16 GiB unified memory, PyTorch2.4.1,
MPS available and CUDA unavailable.

The runner's explicitly selected 240-second operational limit expired. The
first 100-sample batch completed in 178.74 seconds; no eligible query had yet
been recorded. The trace has 100 rows, the probe JSONL has zero rows, and
`query_budget_complete=false`. The runner did not start either Pilot B cell.
An incomplete prefix with no eligible queries cannot establish absence of
headroom and is not a NO-GO result. This timeout does not prove a deadlock or
that a longer MPS run would fail.

Artifacts remain under:
`evidence/oracle-support-utility-mps-final/oracle-support-a-mps-seed0/`.
The parent runner output is [mps-final-execution.log](mps-final-execution.log).
Model digest: `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.
Dataset root digest: `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc`.
Fast verification does not rehash every dataset file.

Earlier exploratory attempts were interrupted or hit the MPS Half indexing
backward error documented in the review. None is completed scientific evidence.
An earlier process sample observed an autograd/Metal wait; it does not establish
a deadlock. All launched experiment processes have stopped.

Two post-attempt fixes are covered by the final test suite: skip the supervised
backward entirely when no cache can qualify a query, and derive empty-summary
oracle scope from the configured mode. The preserved partial attempt's empty
summary incorrectly says `screened_subset_lower_bound`; its manifest correctly
records exhaustive Pilot A, and its loss/headroom values are null. The artifact
has not been silently rewritten. The current runner archives fresh source hashes.

## Remaining work

Run A to 16 eligible ID queries, then B to 128 eligible queries for each ratio,
and compare measured headroom/random/cosine controls before a research decision.
The ready [CUDA plan](cuda-plan.json) contains three config-locked commands.
A fresh CUDA environment can execute all cells with:

```sh
python scripts/run-oracle-support-utility.py \
  --data-root /path/to/data --evidence-dir /path/to/new/evidence \
  --device cuda --execute
```

Alternatively, use `--device mps` with a longer `--timeout-seconds` budget.
Keep batch100; reducing it changes the requested support-cache visibility.
