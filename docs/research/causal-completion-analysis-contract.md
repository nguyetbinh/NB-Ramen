# Causal Scheduling Completion Analysis Contract

This analysis isolates scheduling with the primary paired comparison:

```text
StructuredAtomicRamen vs CausalRamen
```

The methods must share the same deterministic run identity and method configuration except for method identity. `Ramen` is also required in every requested cell as a diagnostic control, but its configuration is not claimed to be equivalent to the structured pair.

## Required coverage

A completion decision requires all requested method cells, at least three fixed seeds, at least two non-IID stream types, full CIFAR-100-C evidence, at least one natural-domain dataset, and at least two evaluator batch sizes including `B=1`. Missing requested evidence is insufficient; malformed or non-equivalent structured-pair evidence is invalid.

Batch-size deltas are descriptive. Monotonicity is not a decision criterion. At `B=1`, legacy Ramen has no future-within-batch visibility, so any remaining CausalRamen-minus-Ramen difference is evidence of implementation or numerical differences rather than isolated scheduling.

## Default decisions

- `GO`: the mean paired micro-accuracy gain is at least `0.01` and its sample standard deviation is at most `0.02`.
- `WEAK_GO`: the stable mean micro gain is nonnegative but below `0.01`, and at least one paired secondary improvement is meaningful: negative-adaptation rate decreases by at least `0.01`, mean recovery improves by at least one sample, or worst-domain accuracy improves by at least `0.005`.
- `NO_GO`: completion coverage is present but neither positive decision is supported.
- `PILOT`: paired evidence is complete for its requested pilot matrix but completion coverage is not met.
- `INSUFFICIENT`: a requested evidence cell is absent.
- `INVALID`: evidence validation or structured-pair configuration equivalence fails.

The canonical defaults are stored in [`causal-completion-go-no-go.json`](../../cfg/research/causal-completion-go-no-go.json). CLI exit status is `0` only for `GO` or `WEAK_GO`, `1` for `NO_GO`, `PILOT`, or `INSUFFICIENT`, and `2` for `INVALID`.
