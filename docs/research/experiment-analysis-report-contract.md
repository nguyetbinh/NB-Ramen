# Experiment analysis report contract

`python -m src.evaluation.experiment_analysis` evaluates completed NB-Ramen
evidence after the experiment matrix has written it. It does not execute a
model, write experiment evidence, or make evaluation labels available to a
method.

The command requires an evidence directory, the selected datasets, streams,
seeds, and a JSON thresholds file. `--device`, `--data-root`,
`--max-eval-samples`, and `--artifact-provenance` must match the matrix
invocation so backend, canonical dataset root, cost-limited, and
exact-provenance run identities are reconstructed exactly.
For cost-limited pilots with a nondefault stream block size, pass the same
`--stream-block-size`; it reconstructs the corresponding `blk-N` run IDs.
The command calls `validate_completed_run` for every run. A malformed, incomplete,
foreign, or tampered run yields canonical JSON with `status: invalid_evidence`
and exit code 2. A valid report exits 0 only for `go`; `no_go` and
`insufficient_evidence` exit 1.

The thresholds JSON must contain only these numeric fields:

- `minimum_repeats`, `max_accuracy_std`
- `structured_degradation_min`, `oracle_recovery_min`, `router_closure_min`
- `natural_domain_gain_min`
- `max_memory_ratio`, `max_forward_latency_ratio`
- `min_routing_accuracy_association`, `max_class_context_nmi`

Phase 02 predeclares its thresholds in
[`cfg/research/phase02-go-no-go.json`](../../cfg/research/phase02-go-no-go.json).
Accuracy values use the `[0,1]` scale. The gate requires three seeds with at
most one percentage point sample standard deviation, at least one percentage
point of Ramen degradation on a non-IID stream, at least one percentage point
of oracle recovery, and at least 25% closure of the positive oracle gap. A
natural-domain gain must be at least 0.5 percentage points. LatentRamen may use
at most 1.10x peak device memory and 1.25x total forward latency relative to
Ramen. Routing quality must have at least a weak positive (`r >= 0.20`)
association with accuracy gain, while context-vs-class NMI must remain below
or equal to `0.80`. These choices are fixed before benchmark evidence is
examined; changing them requires a new named threshold file and report.

The canonical report groups raw validated runs by
`dataset/stream_mode/seed/method`, then aggregates across seeds with count,
mean, sample standard deviation, and normal-approximation 95% CI for accuracy,
worst-domain accuracy, recovery samples, negative adaptation, total forward
latency, throughput, method memory, and routing NMI. It also records
LatentRamen comparisons against Ramen, NoAdapt, and OracleLatentRamen,
including oracle-gap closure.

`class_recovery` reports NMI between inferred contexts and ground-truth class
from the validated trace. High values are a class-clustering warning, not a
success metric. Routing/accuracy association is Pearson correlation between
LatentRamen routing NMI and its paired accuracy gain over Ramen.

Equal-memory comparison uses only same-source device peak evidence:
`peak_device_memory_bytes` when present, otherwise matching
`device_memory.kind` peak values. It never ratios logical support-memory bytes
against device memory, and retains `method_memory` only as a diagnostic.
The structured-degradation criterion uses the maximum observed Ramen
degradation across non-IID streams, matching the roadmap's “at least some"
non-stationary streams requirement. Oracle recovery, router closure, and
natural-domain gain are likewise calculated only from non-IID comparisons;
IID-mixed evidence cannot satisfy or dilute those gates.
Completeness is evaluated across every dataset/stream/seed cell scheduled for
the four paired methods (NoAdapt, Ramen, OracleLatentRamen, and LatentRamen).
Every relevant cell must supply its paired accuracy values; every Latent/Ramen
cell must supply comparable device-memory peaks and total-forward latency; and
every Latent cell must supply routing and class-context evidence. The
structured semantic, oracle, router, and natural-domain criteria require all
of their applicable non-IID cells. A missing control or one unavailable seed
makes the affected criterion `insufficient_evidence`; available seeds are never
silently used as a smaller subset.
Retrieval-only latency remains explicitly unavailable because the evidence
contract intentionally does not isolate it. The report gates the maximum
LatentRamen context-vs-class NMI against `max_class_context_nmi` to reject
routers that mainly cluster classes. Missing values, too few repeats, absent
paired controls, zero oracle gap, or an unidentifiable correlation produce
`insufficient_evidence`; the evaluator never infers a threshold or substitutes
a pass. Trace-derived class recovery is revalidated after reading so a replaced
trace cannot be reported as validated evidence.
