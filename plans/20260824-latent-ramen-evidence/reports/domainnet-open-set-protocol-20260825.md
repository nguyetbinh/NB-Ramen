# DomainNet semantic open-set protocol

Date: 2026-08-25

## Delivered

`src/datasets/open_set.py` now provides `OpenSetDomainNet` and a versioned,
name-based split recipe at `cfg/research/open-set-domainnet-split-v1.json`.
The recipe fixes a 276-known / 69-unknown partition for DomainNet's expected
345 classes.  Since this repository discovers DomainNet classes from
ImageFolder directories, it deliberately does not copy an unverified list of
345 class names into source control.  Instead, it deterministically ranks the
validated class names with the pinned `sha256-name-rank-v1` recipe.

Materialization validates the exact class count and unique non-empty names,
then returns the complete disjoint known/unknown name and original-ID
partitions.  It records a taxonomy SHA-256 and split fingerprint, which future
run-manifest integration must bind and verify.  A changed local taxonomy or
split recipe therefore cannot silently be treated as the same protocol.

The wrapper reduces only `classes` / `num_classes` for CLIP prompt and model
vocabulary construction.  It preserves each source ImageFolder example and
its original label.  Its wrapped domains expose evaluator-only
`original_label`, `known_label_or_minus_one`, and `is_ood` metadata; unknown
labels are never remapped into the model label space.

## Validation

Initial dependency-light command:

```text
python -m pytest tests/test_open_set_domainnet.py tests/test_open_set.py -q
```

Result: `8 passed in 0.06s` on Python 3.12.9 / pytest 8.2.2.

## Direct-evaluation integration

`--dataset DomainNet --open_set --known_class_split
open-set-domainnet-name-rank-v1` now selects `OpenSetDomainNet`.  The direct
CLI accepts this protocol alongside the unchanged CIFAR-100-C protocol and
resolves the split recipe path explicitly.  The generic open-set stream
builder copies optional dataset-provided `open_set_split_fingerprint` and
`open_set_taxonomy_sha256` values into its fingerprinted `open_set` metadata;
they therefore flow into both `stream.json` and the existing run manifest's
dataset metadata without modifying the evidence schema.

Pinned-environment command:

```text
/Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest \
  tests/test_open_set_domainnet.py tests/test_open_set.py tests/test_evaluation_budget.py
```

Result: `Ran 18 tests ... OK` on Python 3.11.16.  This environment matches the
repository's pinned Python-major version and imports the full runtime.  It does
not install pytest, so the standard-library runner was used.  All DomainNet
tests use toy in-test ImageFolder-like datasets and do not require real data.

`python -m compileall -q src/datasets/open_set.py` and `git diff --check` also
completed successfully.

## Deliberately deferred integration

This work does not alter the evidence schema, experiment runtime matrix, or
CIFAR-100-C behavior.  The direct main/stream integration is complete.  The
experiment matrix remains future work by scope: it must explicitly schedule
the DomainNet protocol and preserve the new protocol identifiers in any
matrix-specific manifest validation before it can run this secondary benchmark.
