"""Exact replay of the repository's historical mixed-domain shuffle.

The old evaluator used ``DataLoader(dataset, shuffle=True, num_workers=0)``
without supplying a generator.  This module reproduces that algorithm from
the process-global Torch RNG state at the call site, including its two global
seed draws.  It is intentionally a historical-parity tool, not a method-
independent research stream.
"""

from __future__ import annotations

import torch

from .builders import StreamDataset, _datasets_from


def build_legacy_torch_iid_stream(multi_datasets, seed: int) -> StreamDataset:
    """Replay historical ``DataLoader(..., shuffle=True, generator=None)`` order.

    This must be called at the point where the historical DataLoader iterator
    would have been created: after method construction and before its first
    forward.  DataLoader first draws its worker base seed from the global
    generator, then ``RandomSampler`` draws another global seed and uses a
    newly seeded local generator for ``randperm``.  Both global draws happen
    here, so the resulting order and subsequent global RNG state exactly match
    the historical evaluator.

    ``seed`` records the run's initial seed; it does not by itself determine
    the stream because method construction may already have consumed global
    RNG.  The exported stream fingerprint is the authoritative pairing key.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    datasets, lengths = _datasets_from(multi_datasets)
    # Match _BaseDataLoaderIter's base-seed draw, followed by RandomSampler's
    # generator=None branch.  RandomSampler then performs randperm on its own
    # freshly seeded generator rather than on the global generator.
    base_seed = int(
        torch.empty((), dtype=torch.int64).random_().item()
    )
    sampler_seed = int(
        torch.empty((), dtype=torch.int64).random_().item()
    )
    sampler_generator = torch.Generator().manual_seed(sampler_seed)
    total = sum(lengths)
    permutation = torch.randperm(total, generator=sampler_generator).tolist()

    flat_references = [
        (domain_idx, sample_idx)
        for domain_idx, length in enumerate(lengths)
        for sample_idx in range(length)
    ]
    references = [flat_references[index] for index in permutation]
    environments = getattr(multi_datasets, "environments", None)
    return StreamDataset(
        datasets,
        references,
        {
            "format_version": 1,
            "mode": "legacy_torch_iid_mixed_historical_parity",
            "seed": seed,
            "domain_lengths": list(lengths),
            "domain_names": list(environments) if environments is not None else None,
            "parameters": {
                "sampler": "torch.utils.data.RandomSampler",
                "torch_version": torch.__version__,
                "generator_source": "process_global_state_after_method_construction",
                "replay_contract": (
                    "exact DataLoader(generator=None) parity at call site; consumes "
                    "the historical two global seed draws"
                ),
                "global_rng_draws_consumed": 2,
                "pairing_contract": (
                    "method construction may change order; exported fingerprint "
                    "is authoritative"
                ),
                "dataloader_base_seed": base_seed,
                "random_sampler_seed": sampler_seed,
                "random_sampler_generator": "new_local_generator_seeded_from_global_draw",
                "concatenation_order": "domain_then_sample",
            },
        },
    )
