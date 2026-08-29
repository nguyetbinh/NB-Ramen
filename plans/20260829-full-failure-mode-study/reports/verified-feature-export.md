# Verified feature exporter

Status: implemented

`src.evaluation.verified_feature_export` decodes query features only from a completed,
checksum-validated `replay_v1` sidecar. It binds each feature to the immutable evaluator
trace, requires full trace coverage, and derives its feature and artifact metadata from the
verified run. The exporter accepts only `manifest.args.analysis_role: analysis`; missing,
legacy, or final-evaluation roles are rejected.

Train/validation/test roles are deterministic hashes of immutable non-label sample identity,
the split seed, and immutable artifact provenance. Evaluator labels and domains do not
participate in split assignment or the evaluated method. Each query must also bind exactly to
the validated trace failure-analysis item ID and to the feature-bearing item's segment,
timestep, and evaluator sample identity.

The verified exporter and probe pipeline completed for both MPS block seeds.
Feature-to-domain test accuracy was `1.0` for both seeds, while
feature-to-class test accuracy was `0.0` and `0.1818`. The bounded 64-sample
class-conditioned probes were mostly insufficient (2 computed classes per
seed), so the result demonstrates strong domain decodability but does not by
itself establish class-conditional invariance failure.
