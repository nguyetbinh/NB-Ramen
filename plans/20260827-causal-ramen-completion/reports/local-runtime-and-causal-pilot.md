# Local runtime and causal pilot

Date: 2026-08-27 (Asia/Ho_Chi_Minh)

## Environment and preflight

- Working tree: branch `causal-ramen-completion`; pre-existing source/config/test/docs changes were preserved and no source, config, or test files were modified by this runtime pass.
- Python: `/Users/admin/miniconda3/envs/nb-ramen/bin/python` (Python 3.11.16), with `PYTHONPATH=src`.
- Deep preflight command:
  `PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m runtime.preflight --data-root /Users/admin/data --dataset CIFAR100C --deep --json`
- Result: exit 0, `valid=true`; all 15 CIFAR100C corruption arrays were `uint8`, shape `[50000,32,32,3]`; labels were `uint8`, shape `[50000]`, bounds 0..99, and deep semantic checks passed.
- Artifact: [cifar100c-deep-preflight.json](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cifar100c-deep-preflight.json)

## Unit tests

Command: `PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest discover -s tests -p 'test*.py'`

Result: exit 0; 236 tests ran and passed (`OK`), wall time 2.93s including the shell wrapper. Output: [full-unit-tests.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/full-unit-tests.log).

## CPU mechanics evidence

Both commands used dataset `CIFAR100C`, stream `block`, methods `Ramen StructuredAtomicRamen CausalRamen`, seed 0, CPU, `/Users/admin/data`, `cfg/smoke`, `--artifact-provenance fast`, `--max-eval-samples 4`, and `--stream-block-size 4`, with the requested evidence root.

- B=1 exact command: `env PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m runtime.experiment_matrix --dataset CIFAR100C --stream block --method Ramen --method StructuredAtomicRamen --method CausalRamen --seed 0 --device cpu --data-root /Users/admin/data --config-dir cfg/smoke --artifact-provenance fast --max-eval-samples 4 --stream-block-size 4 --batch-size 1 --evidence-dir /Users/admin/Documents/NB-Ramen/evidence/causal-ramen-completion-cpu-smoke --execute`; wall time 37.77s. NoAdapt completed with run ID `cifar100c-block-seed-0-noadapt-dev-cpu-n4-bs-1-blk-4-cfg-98584622153d-prov-fast-data-6dbea801cbad`. Ramen then failed before producing a valid summary: `RuntimeError: "cdist" not implemented for 'Half'` at `src/methods/Ramen.py:71`. StructuredAtomicRamen and CausalRamen were not launched.
- B=4 exact command: `env PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m runtime.experiment_matrix --dataset CIFAR100C --stream block --method Ramen --method StructuredAtomicRamen --method CausalRamen --seed 0 --device cpu --data-root /Users/admin/data --config-dir cfg/smoke --artifact-provenance fast --max-eval-samples 4 --stream-block-size 4 --batch-size 4 --evidence-dir /Users/admin/Documents/NB-Ramen/evidence/causal-ramen-completion-cpu-smoke --execute`; wall time 32.33s. NoAdapt completed with run ID `cifar100c-block-seed-0-noadapt-dev-cpu-n4-bs-4-blk-4-cfg-98584622153d-prov-fast-data-6dbea801cbad`. Ramen reproduced the same Half-precision CPU `torch.cdist` failure. StructuredAtomicRamen and CausalRamen were not launched.
- Preserved command logs: [cpu-b1-execute.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cpu-b1-execute.log), [cpu-b4-execute.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cpu-b4-execute.log).
- Strict validation: exact invocations with `--execute --resume` were run for both B=1 and B=4. Both exited 1 and rejected the incomplete Ramen summary (`invalid summary`); logs are [cpu-b1-strict-resume.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cpu-b1-strict-resume.log) and [cpu-b4-strict-resume.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cpu-b4-strict-resume.log).

The only completed cells were NoAdapt baselines. At both batch sizes, `num_samples=4`, micro/macro accuracy was 0.25, the only populated domain was `pixelate` at 0.25, and the stream fingerprint was `d9dd816b694e5dc8f187cec82fcd85899843fd135b89aaecd45e9615e04c37ba`. No adapted accuracy, latency, memory, or causal completion delta exists; reporting one would be invalid.

## Causal completion analysis

Exact command: `env PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m evaluation.causal_completion_analysis --evidence-dir /Users/admin/Documents/NB-Ramen/evidence/causal-ramen-completion-cpu-smoke --config-dir cfg/smoke --dataset CIFAR100C --stream block --seed 0 --batch-size 1 --batch-size 4 --device cpu --data-root /Users/admin/data --max-eval-samples 4 --stream-block-size 4 --artifact-provenance fast`.

Result: exit 1, decision `INSUFFICIENT`, because all six requested method/batch cells failed strict validation (invalid Ramen summaries or missing StructuredAtomicRamen/CausalRamen directories). This is not a completion PILOT: the intended paired completion coverage is absent. Canonical output: [cpu-causal-completion.json](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cpu-causal-completion.json); stderr/status: [cpu-causal-completion.status](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cpu-causal-completion.status).

## MPS and CUDA gates

The bounded MPS matrix was not run because the CPU gate failed. Therefore no MPS evidence or MPS causal analysis was claimed or fabricated.

Exact local probe: PyTorch 2.4.1 on `macOS-15.7.7-arm64-arm-64bit`; `torch.version.cuda=null`, `torch.cuda.is_available()=false`, CUDA device count 0, and MPS availability true. MPS is not CUDA. Probe: [cuda-probe.json](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/cuda-probe.json).

DomainNet was not run: the requested CPU gate failed, and no verified local DomainNet completion evidence was available. Final CUDA/DomainNet completion remains blocked on a verified NVIDIA CUDA runner and the required DomainNet data/evidence path.

## Post-fix runtime execution (v2)

The reviewed CPU half-precision fix was present before this pass. The original failed evidence above was retained unchanged. No source, config, or test files were modified during this pass.

### Tests

- Focused command: `env PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest tests.test_ramen_cpu_half tests.test_ramen_memory_bytes`; exit 0, 4 tests passed. Log: [post-fix-focused-tests.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-focused-tests.log).
- Full command: `env PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest discover -s tests -p 'test*.py'`; exit 0, 242 tests passed. Log: [post-fix-full-unit-tests.log](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-full-unit-tests.log).

### CPU v2

The exact B=1 and B=4 matrix commands used the prior CPU identity, with only the evidence root changed to `/Users/admin/Documents/NB-Ramen/evidence/causal-ramen-completion-cpu-smoke-v2`. Both matrix commands exited 0; both exact `--execute --resume` commands exited 0 and skipped all four strictly validated cells. Logs: [B=1 execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-cpu-v2-b1-execute.log), [B=4 execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-cpu-v2-b4-execute.log), [B=1 resume](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-cpu-v2-b1-resume.log), [B=4 resume](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-cpu-v2-b4-resume.log). Per-process wall-clock wrappers were not included in these invocations; exact recorded forward totals are in the summaries and analyzer JSON.

The CPU run IDs were:

- B=1: `...noadapt...bs-1...98584622153d...`, `...ramen...bs-1...7098bd868b96...`, `...structuredatomicramen...bs-1...e238b33d10cb...`, `...causalramen...bs-1...e238b33d10cb...`.
- B=4: `...noadapt...bs-4...98584622153d...`, `...ramen...bs-4...7098bd868b96...`, `...structuredatomicramen...bs-4...e238b33d10cb...`, `...causalramen...bs-4...e238b33d10cb...`.

CPU method metrics (`micro / macro / worst`, forward total ms, retained bytes) were:

| B | NoAdapt | Ramen | StructuredAtomicRamen | CausalRamen |
|---|---|---|---|---|
| 1 | 0.25 / 0.25 / 0.25; 314.611; n/a | 0.25 / 0.25 / 0.25; 1195.017; 323600 | 0.25 / 0.25 / 0.25; 1211.821; 647264 | 0.25 / 0.25 / 0.25; 1248.704; 647264 |
| 4 | 0.25 / 0.25 / 0.25; 273.307; n/a | 0.50 / 0.50 / 0.50; 934.216; 323600 | 0.50 / 0.50 / 0.50; 977.628; 647264 | 0.50 / 0.50 / 0.50; 1034.540; 647264 |

CPU causal analyzer: [post-fix-cpu-v2-causal-completion.json](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-cpu-v2-causal-completion.json), exit 1, `PILOT`. Causal-minus-atomic deltas for B=1/B=4 respectively were micro 0.0/0.0, macro 0.0/0.0, worst 0.0/0.0, forward +36.883/+56.912 ms, memory 0/0 bytes, and negative-adaptation 0.0/0.0. Causal-minus-legacy deltas were micro 0.0/0.0, macro 0.0/0.0, worst 0.0/0.0, forward +53.688/+100.325 ms, memory +323664/+323664 bytes, and negative-adaptation 0.0/0.0. The analyzer’s PILOT reason is that completion coverage is valid but full CIFAR100C, natural-domain, three-seed, and two-stream requirements are not met.

### MPS v2

The requested MPS root existed as an empty directory, so it was not overwritten. Fresh root used: `/Users/admin/Documents/NB-Ramen/evidence/causal-ramen-completion-mps-block-n64-v2`. Exact B=1 and B=8 matrices used CIFAR100C/block, seed 0, `cfg`, fast provenance, n=64, block size 8. Both matrix commands exited 0; both exact `--execute --resume` validations exited 0 and skipped all four cells. Logs: [B=1 execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-b1-execute.log), [B=8 execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-b8-execute.log), [B=1 resume](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-b1-resume.log), [B=8 resume](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-b8-resume.log). Full run IDs are preserved in those logs and the canonical analyzer JSON. Per-process wall-clock wrappers were not included; exact forward totals are recorded below and in the summaries.

MPS method metrics (`micro / macro / worst`, forward total ms, retained bytes) were:

| B | NoAdapt | Ramen | StructuredAtomicRamen | CausalRamen |
|---|---|---|---|---|
| 1 | 0.25 / 0.229167 / 0.0; 2475.484; n/a | 0.0625 / 0.0625 / 0.0; 10615.006; 5177600 | 0.0625 / 0.0625 / 0.0; 8702.052; 5178880 | 0.0625 / 0.0625 / 0.0; 8189.321; 5178880 |
| 8 | 0.25 / 0.229167 / 0.0; 1884.937; n/a | 0.265625 / 0.25 / 0.0; 10463.798; 5177600 | 0.265625 / 0.25 / 0.0; 9449.925; 5178880 | 0.265625 / 0.25 / 0.0; 8625.174; 5178880 |

MPS causal analyzer: [post-fix-mps-v2-causal-completion.json](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-causal-completion.json), exit 1, `PILOT`. Causal-minus-atomic deltas for B=1/B=8 respectively were micro 0.0/0.0, macro 0.0/0.0, worst 0.0/0.0, forward -512.731/-824.751 ms, memory 0/0 bytes, and negative-adaptation 0.0/0.0. Causal-minus-legacy deltas were micro 0.0/0.0, macro 0.0/0.0, worst 0.0/0.0, forward -2425.685/-1838.625 ms, memory +1280/+1280 bytes, and negative-adaptation 0.0/0.0. Status: [post-fix-mps-v2-causal-completion.status](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-causal-completion.status).

MPS evidence is local Apple MPS evidence only; it is not CUDA evidence. The CUDA blocker remains unchanged: this host reports PyTorch 2.4.1, `torch.version.cuda=null`, `cuda_is_available=false`, and zero CUDA devices. Full-CIFAR100C/natural-domain conclusions remain blocked by the analyzer’s declared coverage requirements and a verified NVIDIA/DomainNet runner.

## Post-fix MPS batch-size sensitivity

The already validated MPS v2 root was extended in place only for new cells B=2, 5, 10, 20, 50, and 100; B=1 and B=8 artifacts were preserved. Each exact matrix used CIFAR100C/block, seed 0, MPS, `/Users/admin/data`, `cfg`, fast provenance, `--max-eval-samples 64`, and `--stream-block-size 8`. All six matrix commands exited 0, and every corresponding exact `--execute --resume` validation exited 0. The execute logs include true exit codes and `/usr/bin/time -p` durations:

| B | execute wall time | resume | run IDs |
|---|---:|---:|---|
| 2 | 73.07s | 0 | full IDs in [execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b2-execute.log) |
| 5 | 73.67s | 0 | full IDs in [execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b5-execute.log) |
| 10 | 77.47s | 0 | full IDs in [execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b10-execute.log) |
| 20 | 80.21s | 0 | full IDs in [execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b20-execute.log) |
| 50 | 129.35s | 0 | full IDs in [execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b50-execute.log) |
| 100 | 163.45s | 0 | full IDs in [execute](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b100-execute.log) |

Resume logs: [B=2](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b2-resume.log), [B=5](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b5-resume.log), [B=10](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b10-resume.log), [B=20](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b20-resume.log), [B=50](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b50-resume.log), and [B=100](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-b100-resume.log).

For the canonical sensitivity set B=1, 2, 5, 10, 20, 50, 100, the per-method metrics are shown as `micro / macro / worst; forward total ms; retained bytes`:

| B | NoAdapt | Ramen | StructuredAtomicRamen | CausalRamen |
|---|---|---|---|---|
| 1 | 0.250 / 0.229 / 0.000; 2475.484; n/a | 0.0625 / 0.0625 / 0.000; 10615.006; 5177600 | 0.0625 / 0.0625 / 0.000; 8702.052; 5178880 | 0.0625 / 0.0625 / 0.000; 8189.321; 5178880 |
| 2 | 0.250 / 0.229 / 0.000; 2088.967; n/a | 0.265625 / 0.250 / 0.000; 9796.988; 5177600 | 0.265625 / 0.250 / 0.000; 8293.260; 5178880 | 0.265625 / 0.250 / 0.000; 8260.923; 5178880 |
| 5 | 0.250 / 0.229 / 0.000; 2497.446; n/a | 0.28125 / 0.271 / 0.125; 9804.409; 5177600 | 0.28125 / 0.271 / 0.125; 8832.406; 5178880 | 0.265625 / 0.250 / 0.000; 8565.615; 5178880 |
| 10 | 0.250 / 0.229 / 0.000; 1986.019; n/a | 0.28125 / 0.271 / 0.125; 11927.944; 5177600 | 0.28125 / 0.271 / 0.125; 9127.707; 5178880 | 0.265625 / 0.250 / 0.000; 8768.926; 5178880 |
| 20 | 0.250 / 0.229 / 0.000; 2030.393; n/a | 0.28125 / 0.271 / 0.125; 9100.835; 5177600 | 0.28125 / 0.271 / 0.125; 8592.866; 5178880 | 0.265625 / 0.250 / 0.000; 9057.072; 5178880 |
| 50 | 0.250 / 0.229 / 0.000; 2006.123; n/a | 0.296875 / 0.271 / 0.000; 48164.506; 5177600 | 0.296875 / 0.271 / 0.000; 16386.726; 5178880 | 0.265625 / 0.250 / 0.000; 10685.825; 5178880 |
| 100 | 0.250 / 0.229 / 0.000; 1928.722; n/a | 0.265625 / 0.240 / 0.000; 64275.477; 5177600 | 0.265625 / 0.240 / 0.000; 29084.923; 5178880 | 0.265625 / 0.250 / 0.000; 14146.494; 5178880 |

Canonical analyzer: [post-fix-mps-v2-batch-sensitivity.json](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity.json), stderr [post-fix-mps-v2-batch-sensitivity.stderr](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity.stderr), status [post-fix-mps-v2-batch-sensitivity.status](/Users/admin/Documents/NB-Ramen/plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity.status). It exited 1 with `PILOT`; all seven batch sizes were observed and strictly validated. The aggregate mean micro gain was -0.0111607 (std 0.0118114), and the mean worst-domain gain was -0.0535714. Causal-minus-Atomic deltas by B (micro, macro, worst, forward ms, retained bytes) were: B=1 `(0, 0, 0, -512.731, 0)`, B=2 `(0, 0, 0, -32.337, 0)`, B=5 `(-0.015625, -0.020833, -0.125, -266.791, 0)`, B=10 `(-0.015625, -0.020833, -0.125, -358.781, 0)`, B=20 `(-0.015625, -0.020833, -0.125, +464.206, 0)`, B=50 `(-0.03125, -0.020833, 0, -5700.901, 0)`, and B=100 `(0, +0.010417, 0, -14938.429, 0)`. Causal-minus-Legacy deltas (micro, macro, worst, forward ms, retained bytes) were: B=1 `(0, 0, 0, -2425.685, +1280)`, B=2 `(0, 0, 0, -1536.065, +1280)`, B=5 `(-0.015625, -0.020833, -0.125, -1238.794, +1280)`, B=10 `(-0.015625, -0.020833, -0.125, -3159.018, +1280)`, B=20 `(-0.015625, -0.020833, -0.125, -43.764, +1280)`, B=50 `(-0.03125, -0.020833, 0, -37478.681, +1280)`, and B=100 `(0, +0.010417, 0, -50128.983, +1280)`.

Scientific gate: StructuredAtomicRamen and CausalRamen are equal at B=1 and B=2, but do not remain equal across the full tested sensitivity set; CausalRamen is lower on accuracy at B=5/10/20/50 and has a small macro gain at B=100. This bounded seed-0, single-stream pilot still does not justify a three-seed/full-CUDA escalation yet because full-CIFAR100C, natural-domain, multi-seed, and multi-stream completion requirements remain absent. This is a gate/pilot statement only, not a final `NO_GO`.
