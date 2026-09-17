# Vectorized ranking review — 2026-09-07

Read-only review found no blocking semantic issues. Candidate weights, class addition order, accumulator dtype, legal swap enumeration, and scalar dot-product ranking are preserved. CPU/MPS tests compare every vectorized candidate against serial aggregation exactly for half/float cache values. Tests use float accumulators; actual production data also receives runtime parity checks on every candidate of the first eligible query per batch.

Validation: 367 tests run, 362 passed, 5 skipped. Source snapshots for Pilot B are saved at launch. Pilot A source was reconstructed by reversing this patch and verified byte-for-byte against every original launch SHA-256; its audit now reads that archived source so subsequent code changes do not invalidate provenance.

The first B attempt was interrupted during ranking, before completed diagnostic query rows, after observing excessive scalar aggregation overhead. Its evidence remains separate from the vectorized run.
