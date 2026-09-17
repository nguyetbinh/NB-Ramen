# Oracle support utility diagnostic

Implemented 2026-09-07. This diagnostic asks whether changing cached support
identity within a prediction-balanced local neighborhood improves an ID query.
It is excluded from the canonical 252-run matrix and is not deployable TTA.

The returned predictions and retained gradients follow ordinary Ramen. The
probe preserves cache admission, batch=100 visibility, top-k=5, capacity=750,
beta=5, lr=.01 and state-free SignSGD. Supervised known labels and domain/OOD
metadata enter through an evaluator-only, single-use hook. Known labels are
never cached. OOD rows use label -1 and are excluded from supervised loss and
utility search. Missing, stale, inconsistent or mismatched context fails closed.

For each eligible ID query, the probe retrieves candidates at the same cache
snapshot, after the entire current batch was admitted. A query is eligible
when at least one class contains m=10 entries. All classes retain their normal
min(k, size) support counts; classes with k < size < m can contribute their
available extras, and classes with at most k entries remain unchanged. Each swap exchanges one selected support for an extra
within the same pseudo-class. Entropy and distance weights are recomputed,
with the baseline's dtype, class accumulation order and active-class divisor.

Top-k is retrieved separately from top-m because ties can produce different
indices. Extras come from the top-m neighborhood excluding the exact baseline
indices; equal-distance boundary ties retain baseline membership. A strict
aggregation equality check guards the no-swap baseline.

The evaluator computes a supervised per-query gradient at pretrained parameters.
It ranks swaps with lr times its dot product with the change in SignSGD direction.
Exact verification resets parameters, installs the candidate cached gradient and
forwards the original full batch. Identical sign directions reuse their exact
loss/margin/prediction because the optimizer is stateless. This removes redundant
forwards without approximating the result. Each temporary update is reset,
including on exceptions. Gradients are supplied from cache, not recomputed from
support examples.

Pilot A evaluates every legal swap for 16 ID queries at OOD=.5. Pilot B evaluates
first-order best/worst/random plus the gradient-cosine preference for 128 ID
queries at each of OOD=0 and .5. Duplicate choices are evaluated once. The cosine
baseline maximizes incoming-minus-outgoing supervised gradient cosine. Feature
similarity retains the original top-k (no swap). The no-swap fallback is included
in both oracles; raw best forced swap and all checked swaps remain in the output.

Pilot A reports exhaustive one-swap headroom. Pilot B reports a **lower bound on
that headroom from a screened subset**, not the exact best over all swaps.
Pooled Spearman uses average ranks for ties and is null when undefined. Screened
correlation has selection bias. No quantitative GO threshold is inferred from
the source plan's qualitative criteria; compare both OOD ratios and random
controls before making the research decision.

## Running

Use the repository Python environment with PyTorch/CLIP. Plan all three cells:

```sh
python scripts/run-oracle-support-utility.py \
  --data-root /path/to/data --evidence-dir /path/to/new/evidence --device cuda
```

Add `--execute` to execute A, B-null and B-open sequentially. The runner locks
config bytes, archives source hashes, verifies official model/data provenance,
and stops on failure, timeout or an unmet eligible-query budget. It defaults to
600 stream samples (six full legacy batches), source budget 400/domain,
block=64, seed=0, split v1, CLIP ViT-B/16 and a one-hour timeout per cell.
`--max-eval-samples` can extend the stream by complete 100-row batches if the
eligibility target is not reached. Use a fresh evidence directory for retries.
`--timeout-seconds` controls the operational limit; interrupted runs remain
incomplete and cannot support a final headroom claim.

The manifest and ordinary trace describe the unchanged Ramen output. Separate
`oracle-support-queries.jsonl` and `oracle-support-summary.json` carry the probe
results and are refreshed after every completed query and batch. Rows contain
raw verified swaps, support provenance, distance/entropy/cosine, loss, margin,
accuracy, candidate counts and actual exact-forward counts. The summary includes
quantiles, headroom rate, accuracy transitions, net corrections, margin gain,
Spearman, baseline gains and replacement rates. Replacement-rate denominators
include only queries where a positive-utility swap was selected. Probe timing
includes diagnostic work and must not be compared to deployable-method latency.

The ordinary trace's `memory_bytes` remains Ramen's retained cache tensor payload;
it excludes evaluator provenance metadata and temporary search workspaces.

For CLIP ViT, exact trials now evaluate one direction for each eligible query
concurrently on the original full batch and original row positions. Per-sample
LayerNorm parameters are independent. Pilot A retains all legal swap records;
other backbones retain serial trials. The runtime first checks temporary baseline
installation against original Ramen logits, then checks the first two concurrently
updated rows against isolated full-batch trials with exact tensor equality.
It restores normalization row counts directly instead of a discarded feature
forward. See the [verification review](../../plans/20260907-oracle-support-utility/reports/grouped-verification-review.md).


The first completed [Pilot A result](../../plans/20260907-oracle-support-utility/reports/pilot-a-completed.md)
contains 16 eligible queries and all 1,760 swaps on MPS. Its diagnostic target is
complete, while its generic 600-sample evaluator was stopped after 200 committed
trace rows once the target was reached. Do not classify it as a full-stream or
600-sample benchmark. The report links an independent raw-artifact audit and
records the single-corruption scope and unchanged accuracy.


[Pilot B is complete](../../plans/20260907-oracle-support-utility/reports/pilot-b-completed.md):
128 screened queries at each OOD ratio, with independently audited raw artifacts.
Mean loss gain is 0.046677 at OOD=0 and 0.331788 at OOD=.5; each cell gains
two correct predictions. Their query/domain/cache histories differ, so this is
not a causal estimate of OOD contamination. Both stop after diagnostic completion
(200/400 trace rows), not a full 600-sample benchmark.

Swap aggregation now vectorizes independent candidates while preserving class
addition order and scalar dot-product ranking. Every candidate on the first
eligible query of each batch is checked against serial arithmetic. Source snapshots
and the exact A/B overlap audit preserve reproducibility across this optimization.
