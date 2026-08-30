# Ramen Failure Decomposition — Execution Plan

Date: 2026-08-30
Branch: `ramen-failure-decomposition`
Base: `causal-ramen-completion@986a6e7cde03f1667fc05510dd9a21a2ef52acb7`

## Goal

Stop proposing mechanisms before identifying the dominant residual failure in Ramen. The study order is fixed as:

```text
baseline reproduction
-> exact outcome decomposition
-> component instrumentation
-> evaluator-only oracle ladder
-> dominant-failure decision
-> smallest targeted method
-> method ablation
-> final experiment
-> final error analysis
```

The methodological reference is Galstyan et al., *Failure Modes of Domain Generalization Algorithms* (CVPR 2022): decompose final error into mechanistically distinct components and do not treat a proxy metric as a task-level success criterion.

## Frozen conclusions from previous branches

These are controls/negative ablations, not active methods to tune on target evidence:

- latent routing: current router collapses; annotated domain identity is not sufficient for better adaptation;
- entropy gate: improves cache/pseudo-label purity but lowers accuracy;
- strict causality: retained as the clean strict-online protocol control; scheduling-only accuracy gain is unsupported by the bounded same-memory comparison;
- retrieval compression: deferred because retrieval was not the dominant profiled bottleneck in the bounded pilot.

## Phase 1 — exact outcome-level decomposition

### Question

Is Ramen weak because it harms predictions that were already correct, or because it fails to rescue predictions that were initially wrong?

### Required paired categories

```text
NoAdapt correct -> Ramen correct : stable_correct
NoAdapt wrong   -> Ramen correct : beneficial_adaptation
NoAdapt correct -> Ramen wrong   : harmful_adaptation
NoAdapt wrong   -> Ramen wrong   : unresolved
```

### Exact identity

For exactly paired traces:

```text
Acc(Ramen) - Acc(NoAdapt)
= beneficial_adaptation_rate - harmful_adaptation_rate
```

### Implemented foundation

`src/evaluation/failure_decomposition.py` provides:

- strict paired trace validation;
- four-way decomposition;
- conditional help/harm rates;
- per-domain decomposition;
- temporal windows;
- scalar mechanism diagnostics stratified by outcome;
- ordered oracle-ladder decomposition with negative/interacting increments preserved;
- trace SHA-256 provenance;
- fail-closed CLI.

CLI examples:

```bash
PYTHONPATH=src python -m evaluation.failure_decomposition outcomes \
  --reference-trace evidence/noadapt/trace.jsonl \
  --adapted-trace evidence/ramen/trace.jsonl \
  --window-size 32 \
  --window-stride 16 \
  --output evidence/failure-analysis/outcomes.json
```

Future compact scalar diagnostics can be attached without changing the outcome analyzer:

```bash
PYTHONPATH=src python -m evaluation.failure_decomposition outcomes \
  --reference-trace evidence/noadapt/trace.jsonl \
  --adapted-trace evidence/ramen-diagnostic/trace.jsonl \
  --mechanism-field gradient_consensus_mean \
  --mechanism-field gradient_pairwise_sign_disagreement \
  --mechanism-field retrieval_active_class_count \
  --output evidence/failure-analysis/mechanism-by-outcome.json
```

## Phase 2 — opt-in compact instrumentation

Do not log full gradients.

Add a diagnostic profile that does not alter support selection or parameter updates. Required per-query fields:

```text
pre-adaptation prediction / entropy
current method item ID
support item IDs
support predicted classes
support distances
support entropies
support recencies
support weights
active support-class count
gradient consensus mean
gradient low-consensus fraction(s), preregistered after pilot
pairwise class-gradient sign disagreement
aggregate gradient norm / zero-sign fraction
```

Evaluator-only joins may append:

```text
support true class
support true domain/corruption
support pseudo-label correctness
ID/OOD status in later open-set experiments
```

Ground-truth joined fields must never be passed back into a deployable method.

### Invariance test

For every instrumentation change, add a test proving that diagnostic `off` and diagnostic `compact_v1` produce identical:

```text
retrieved support IDs
aggregate adaptation gradient
adapted prediction
memory eviction state
```

for the same deterministic toy stream.

## Phase 3 — component/oracle ladder

The failure ladder should be cumulative and ordered. Each step changes one declared stage while keeping previous oracle replacements fixed.

Initial ladder:

```text
R0 Standard Ramen
R1 Oracle memory/candidate availability diagnostic
R2 + Oracle retrieval
R3 + Oracle weighting
R4 + Oracle gradient objective
R5 + Oracle aggregation/compatibility
R6 + Oracle update/step size
```

For each stage report:

```text
accuracy
error
incremental error reduction versus previous stage
beneficial/harmful flip counts versus R0
regression/interactions when an increment is negative
```

Do not force increments to be positive. A negative increment is evidence of interaction or a poor oracle definition and must remain visible.

## Phase 4 — isolate retrieval before redesigning routing

Two separate questions must not be conflated:

```text
A. Does Ramen's feature-similarity proxy fail to recover same-domain support?
B. Even with true-domain support, is domain identity the right adaptation partition?
```

Required comparison:

```text
Ramen retrieval
vs oracle same-domain retrieval
vs oracle gradient-compatible retrieval
```

Interpretation:

```text
OracleDomain >> Ramen
    -> retrieval proxy/headroom exists; routing may be worth revisiting

OracleDomain ~= Ramen and OracleCompatible >> Ramen
    -> domain identity is the wrong objective; do not tune the latent router
```

## Phase 5 — isolate entropy weighting from entropy-gradient objective

Entropy enters Ramen in two logically separate places:

```text
support weighting
and
self-supervised entropy-gradient generation
```

Required controls:

1. same supports + same cached gradients, replace entropy/similarity weights with evaluator-only compatibility weights;
2. same memory/support/weights/aggregation, replace entropy gradients with supervised evaluator-only gradients.

This distinguishes:

```text
reliability-weight proxy failure
from
adaptation-objective failure
```

## Phase 6 — gradient-conflict diagnostic

This is the current highest-value unresolved mechanism question, but it is not yet a method claim.

For active support class `c`, compute a class-local weighted gradient `h_qc`. Because Ramen ultimately uses SignSGD, summarize coordinate-wise sign agreement across active classes without writing the full sign vector to disk.

Primary outcome contrast:

```text
beneficial_adaptation
vs
harmful_adaptation
```

Go/no-go for a consensus mechanism requires all three:

1. harmful queries show more conflict in a stable direction;
2. the association repeats across fixed seeds and at least two structured stream types;
3. an evaluator-only compatibility oracle recovers a material portion of the harmful-update gap.

If these fail, reject ConsensusRamen before implementation.

## Pilot protocol

Use bounded CIFAR-100-C analysis streams first to validate mechanics and metrics. Do not promote MPS pilots to benchmark claims.

Recommended first cells:

```text
Dataset: CIFAR100C
Methods: NoAdapt, Ramen, CausalRamen diagnostic control
Streams: block, recurring
Seeds: 0, 1, 2
Batch size: canonical benchmark batch plus B=1 diagnostic
Prefix: small enough for local mechanics, large enough to contain transitions/recurrence
```

Only after diagnostics are stable should the study escalate to full CIFAR-100-C CUDA and a natural-domain dataset.

## Decision table

The next method is selected by oracle headroom, not preference:

| Dominant recovered error | Research direction |
|---|---|
| memory/candidate availability | retention/capacity/admission |
| retrieval | retrieval/partitioning/routing |
| weighting | compatibility/reliability weighting |
| gradient objective | better self-supervised TTA objective |
| aggregation | consensus/conflict-aware aggregation |
| optimizer/update | step size/optimizer/parameter-subspace control |
| no stage has material headroom | stop mechanism expansion; revisit baseline/model limits |

## Stop rule

```text
No new mechanism without diagnosed and replicated headroom.
```
