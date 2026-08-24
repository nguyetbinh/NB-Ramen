# Artifact provenance

`src.runtime.artifact_provenance` separates offline, exact local inventory
generation from run-safe verification. `generate_cifar100c_provenance` accepts
only the pinned official Zenodo record 3555552 evidence: DOI
`10.5281/zenodo.3555552`, the API content URL, size `2918473216`, and MD5
`11f0ed0f1191edbf9fa23466ae6021d3`. `generate_domainnet_provenance` creates
the equivalent explicit/offline per-image inventory.

The generate CLI accepts this acquisition record through `--acquisition-json`;
for CIFAR-100-C the complete record must equal those pinned official fields.
`verify_official_cifar100c_archive` checks the pinned size and MD5. The generic
`archive_acquisition_record` only checks caller-supplied integrity metadata and
is deliberately not treated as publisher proof.

Both use the canonical sidecar location
`<dataset-root>/.nb-ramen-provenance/<dataset>-v1.json`.  Normal verification
checks the regular, non-symlinked sidecar, canonical relative paths, full file
inventory, and file sizes through non-following descriptors without hashing
large arrays. Passing `exact=True` also streams every file and compares
SHA-256 values. The sidecar is an unsigned canonical local content inventory,
not a signature or immutable publisher manifest. Its own SHA-256 and the
inventory root digest are archived in each verified run manifest.

CLIP verification uses an in-code table of the five supported OpenAI model
URLs and SHA-256 digests. It does not trust importable package `_MODELS`
metadata. Verified runs pass the exact hashed checkpoint pathname to
`clip.load`, then rerun model and dataset verification after loader
construction and require identical reports before writing the manifest.

Fast verification deliberately validates inventory and sizes rather than file
content; exact mode rehashes content. Path identity checks close the direct
symlink replacement window for each file, but traversal is pathname-based:
concurrent mutation of ancestor directories is outside the guarantee. A
concurrent writer can also replace artifacts after the post-load check. Use
read-only artifact mounts or filesystem snapshots for scientific runs.
