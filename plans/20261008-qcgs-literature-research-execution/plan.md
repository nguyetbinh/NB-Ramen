---
title: "QCGS literature reopening → exact objective audit → research decision"
status: in-progress
branch: qcgs-objective-audit
created: 2026-10-08
owner: research integrator
scope: "Thực thi toàn bộ synthesis theo các gate nghiên cứu; các bước sau audit phụ thuộc evidence"
---

# Plan thực thi QCGS literature research synthesis

## 1. Mục tiêu và trạng thái

Đầu vào chính là [qcgs-literature-research-synthesis.md](../../docs/research/qcgs-literature-research-synthesis.md). Workflow theo [research-orchestrator](../../.agents/skills/research-orchestrator/SKILL.md): Literature → Question → Instrument → Research loop → Case review → Paper evidence; mở lại gate trước nếu evidence mới làm mất hiệu lực giả định.

**Câu hỏi trung tâm sau P1:** Trong một query/cache state Ramen cố định, ba objective delta không nhãn đã đăng ký có xếp hạng được các legal one-support swaps theo CE utility của chính query đó hay không? Exact values là information ceiling; chưa phải ranking triển khai được.

**Đích khoa học:** Tách objective alignment khỏi estimator quality; đưa ra quyết định có bằng chứng về ranking, abstention hoặc giới hạn của observation channel. Một STOP hợp lệ có thể hoàn tất một nhánh nghiên cứu; không yêu cầu mọi nhánh tạo ra phương pháp tốt hơn.

Synthesis báo cáo oracle headroom, STOP của single-view entropy-sign và STOP Stage A của MV-target-θ_R. Plan coi đây là đầu vào cần đối chiếu với reports/raw artifacts, chưa mặc nhiên là kết quả đã được xác minh lại trong chu kỳ này. Roadmap cũ không quyết định trạng thái hiện tại.

Plan này đã được commit tại `45ea8f4`; P0–P2, implementation/review/CPU validation và hai CUDA smoke đã hoàn tất theo reports/raw hiện có. Source fix mới nhất là `04abdba` ở coding worktree. Stage A chưa chạy; việc tiếp theo là áp dụng revision resource policy bên dưới, reconcile source/receipts và hoàn tất registry/launch lock tối thiểu để chạy Stage A.

## Execution snapshot — trước implementation validation, 2026-10-08

- P0: branch/commit, runtime, GPU, checkpoint, CIFAR-100-C arrays và thiếu raw artifacts đã được kiểm tra; chi tiết ở [research state](reports/research-state.md).
- P2a: source-only instrument/parity audit hoàn tất ở [instrument audit](reports/instrument-audit.md). Không có code hoặc inference chạy trong task này.
- P1: bounded source-backed map hoàn tất ở [literature verification](reports/literature-verification.md), [gap reassessment](reports/literature-gap-reassessment.md), và [problem map](reports/problem-map.md). Action identifiability đã được nghiên cứu ở mức whole-stream actions; selective adapt/skip và CAS đã được công bố ở ECCV 2026; CATE là neighbor hiện tại cho consensus/calibration. G1/G3 rộng không còn là novelty claims. Query-local same-cache support-action audit vẫn là giả thuyết mở, không phải novelty đã xác lập.
- Historical query exclusion: do không có raw MV registry, đã lập danh sách superset outcome-independent từ toàn bộ bucket Stage-A theo registry fingerprint/namespace đã khóa, gộp với 507 ID cũ: 1.499 ID bị loại, còn 8.501. Danh sách/hashes ở `reports/historical-exclusion-audit.json`; cache-stream exposure vẫn cần phân biệt.
- P2: v1 was committed at `973d81d`; an independent read-only `gpt-6.1-sol`/`low` consistency audit found four ambiguities before any run. The corrected v1.1 protocol SHA-256 is `f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`; findings/dispositions are in [protocol review](reports/protocol-consistency-review.md).
- P3a: bản nháp hiện diện trong workspace chính ở `src/methods/QCGSExactObjectiveAudit.py`, `src/methods/qcgs_exact_objectives.py`, `src/runtime/qcgs_exact_objective_audit.py`, `src/evaluation/qcgs_exact_objective_audit.py`, config và regressions; có integration edits ở `src/main.py`, method exports và registry. Các thay đổi này còn uncommitted tại thời điểm review. Coding worktree `qcgs-exact-objective-code` vẫn tồn tại; không suy ra coder còn active chỉ từ worktree. Cần reconcile bản nháp với protocol v1.1 và chốt source dùng để review/run.
- P3b/P3c: independent registry partition check and pre-run case card are in [case readiness](reports/case-execution-readiness.md): after exclusions, 8,501 base IDs partition to 926 Stage A / 7,575 Stage B under the frozen namespace and current fingerprint. This does not validate stream eligibility or quotas.
- Status review này chỉ đọc source và artifacts, không chạy tests/inference. Tests đã được viết nhưng chưa có validation receipts trong plan; CUDA smoke, scored campaign và executable launch lock chưa được chứng minh hoàn tất. `reports/research-state.md` và `reports/case-execution-readiness.md` là snapshots trước bản nháp implementation; cập nhật chúng bằng receipts khi khép P3.

### Quyết định sau status review — CONTINUE có giới hạn

Implementation vẫn phục vụ synthesis §24: đo exact objective-to-utility alignment trước estimator design. Khoảng 2.190 dòng trong bốn module mới và 686 dòng tests là đầu tư instrument, chưa phải empirical progress. Ưu tiên tiếp theo là khép validity và feasibility của P3, rồi lấy Stage-A evidence.

- Giữ ba objective đã freeze, candidate/update/baseline và scientific gates theo protocol v1.1. P5/P6 vẫn phụ thuộc evidence; không mở thêm objective, learned scorer, HVP, dataset sweep hoặc refactor framework trong P3.
- Stage-B scaffolding đã có; chỉ sửa correctness/protocol defects cần thiết, hoãn mở rộng reporting và tối ưu riêng Stage B tới sau `GO_CONFIRM`.
- Review có mục tiêu, regressions, commit source → CUDA smoke trên IDs riêng → freeze launch lock → Stage A 16 queries/cell. Không tiếp tục hoàn thiện hạ tầng chung trước khi smoke trả lời phép đo có chạy nổi trong budget hay không.
- Success của P3 là instrument có receipts về parity/reset/access/coverage và cost; success của P4 là quyết định raw-derived hợp lệ. Scientific STOP đóng các formulations đã đo; resource shortfall là INCONCLUSIVE, không phải evidence chống hypothesis.

## Execution snapshot — P3 implementation validation tại b33549a (2026-10-08)

- Implementation đã commit trên branch/worktree `qcgs-exact-objective-code`: `b33549ad2e4e35e349a42909f7dbd4e32286ce58`, base `d823bd511da74d838e329d0d2533ea9cec048373`. Source-tree SHA-256 là `116f180b890309cc7a951c9633298053e4c6efa32d565ce96498d81ae2c37e4e` (Git archive tar của `src`); config-bundle SHA-256 là `f2ba05f37e8c8755d7f76ca6ff2d03ee8a1afb33a7f9c821629ad8c5743fd05d`; protocol SHA-256 là `f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`. Commit author/committer dùng Git global identity `[historical Git identity redacted]`.
- 32 exact-objective tests pass; full repository suite pass: 440 tests, 1 skipped. Final targeted `gpt-6.1-sol`/low review trên random-control utility, isolation receipt và direct objective replay regression không còn finding. Dispositions được ghi ở [instrument implementation review](reports/instrument-implementation-review.md).
- Review sửa các lỗi interrupted-run status/JSONL, unpaired action identity, scan suffix continuity, smoke receipt completeness, Stage-B prerequisite/objective binding và random-forced utility recomputation. Formula regression independently replays all three objective calculations on each real scorer baseline/action tensor in the deterministic CPU fixture.
- `action_chunk_size` nhóm query theo action index. Raw receipt nêu rõ GPU isolation sampling scope: action đầu tiên của query chunk đầu, tối đa hai query. CPU parity regression so sánh chunked/serial objective outputs trên mọi query/action của fixture; campaign mặc định dùng chunk 1.
- P3 chưa đóng: chưa chạy scans, CUDA smoke, scored inference hoặc tạo registry/launch lock. Vì vậy peak memory/cost trên CLIP CUDA và smoke cap 60 phút chưa được kiểm chứng.

## Revision hiện hành — bỏ memory cap dự phòng, ưu tiên Stage A (2026-10-08)

**Authorization và lý do:** Người dùng yêu cầu bỏ việc chặn research chỉ vì peak vượt cap dự phòng 7,5 GiB dù smoke chạy xong. Revision này thay execution policy trước scored Stage A, dựa trên cost-only measurements; không dùng supervised utility để sửa objective hoặc scientific gates. Các execution snapshots phía trên là lịch sử, được supersede bởi trạng thái này.

- Reviewed source fix `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93` ở coding worktree đã chạy hai scans 600 rows/cell và hai CUDA smoke 2 queries/cell. Reports ghi 35 focused tests và full suite 443 tests (1 skipped). Smoke có 485 / 850 legal swaps/query, peak 7,785 / 7,794 GiB, tổng elapsed 35:08; cả hai complete, không OOM và không lưu evaluator utility outcomes. Stage A chưa chạy.
- **Bỏ 7,5 GiB làm pass/fail hoặc launch gate.** Peak memory là cost cần đo và báo; không thay bằng memory threshold dự phòng khác. Run complete với peak cao không tự tạo STOP/INCONCLUSIVE hay yêu cầu profiling case mới.
- Giữ ba objectives, candidate/update/baseline, views, quotas, label isolation, parity/reset, Stage-A/B separation và scientific GO/STOP gates. Giữ compute budget Stage A ≤8 GPU-hours/cell, ≤16 tổng; Stage B ≤24 GPU-hours/cell nếu được mở. Profiling và tối ưu runtime/memory có mục tiêu thuộc P3: smoke 35:08 cho bốn queries là lý do đủ để kiểm bottleneck. Estimator và mở rộng hạ tầng chung vẫn phụ thuộc scientific evidence.
- Giữ builder rejection theo policy cũ và mọi raw attempts. Draft handoff trong coding worktree đóng cycle là `INCONCLUSIVE/resource` chỉ vì cap 7,5 GiB không còn là quyết định hiện hành. Câu hỏi khoa học vẫn active; next experiment là Stage A.
- Trước launch, cập nhật resource clause trong protocol/config và registry validator để bỏ threshold 7,5 GiB; ghi revision/hash mới và policy reuse existing smoke receipts. Không sửa raw receipts/config cũ hay mô tả chúng như đã chạy dưới protocol mới. Chỉ kiểm validator/provenance behavior bị ảnh hưởng; không bắt lặp full review/suite/smoke chỉ vì sửa execution policy.
- Reconcile reviewed source fix và receipts từ coding worktree vào checkout dùng để chạy; hoàn tất registry/launch lock tối thiểu với source/config/protocol hashes và unique run IDs, rồi **chạy Stage A 16 queries/cell**. Existing smoke chỉ xác minh workload đã đo; theo dõi actual memory/runtime cho workload dài hơn.
- Actual OOM/crash: giữ failed attempt, ghi failed execution và incomplete evidence; sửa lỗi cụ thể với parity được bảo toàn hoặc chuyển sang máy đủ tài nguyên. Hết compute budget trước đủ coverage là INCONCLUSIVE. Peak cao của run complete tự nó không phải execution failure.

### Tối ưu để giảm thời gian experiment

- Profile một workload cost-only đại diện trên smoke IDs đã loại khỏi scientific registry; tách thời gian/peak của baseline forward/backward, full-batch action trials, view forwards, gradient aggregation và CPU/hash/serialization. Không dùng supervised utility để chọn optimization. Dùng measurements để xác định bottleneck; không suy ra từ tổng elapsed rằng memory là nguyên nhân chạy chậm.
- Ưu tiên sửa cục bộ ở phần chiếm runtime: giảm tính toán lặp, giảm transfers/synchronization không cần thiết, reuse immutable inputs, hoặc batching nếu giữ đúng by-sample parameter/view mapping. Memory reduction có ích nếu giảm overhead hoặc cho phép batching nhanh hơn; không cần đạt cap 7,5 GiB.
- Mỗi thay đổi so before/after trên cùng workload, báo elapsed/throughput và peak memory, kiểm baseline/objective values, selections, reset/cache, legal-action coverage và label isolation. Nếu thay execution path, chạy checks liên quan và CUDA parity/smoke phù hợp; không reuse receipt cũ như chứng cứ cho code mới.
- Giữ mọi legal actions/views/queries và exact objective formulas; không dùng approximation, sample bớt actions hay đổi update để tăng tốc. Optimization hợp lệ không cần mở research case riêng. Sau khi xử lý bottleneck đã đo, freeze source/config và chạy Stage A; không biến tối ưu thành vòng refactor không có điểm kết thúc.

### Execution status — 2026-10-08

- Resource-policy/profile implementation committed at `5f486e4ada84fb5b6a1c4794c3a00dc9015654d5`; execution-report update committed at `f06094431ebe47258fcb8d4be4a1cdee074caf8f`. Protocol v1.2 SHA-256: `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`.
- Focused suite: 36 pass; syntax, protocol JSON and diff checks pass. Fresh targeted review has no remaining findings. The 600-row registry check accepted high smoke peaks but requested 1,200-row scans after OOD=0 was short by 8 Stage-A and 51 Stage-B eligible queries.
- First profile launch attempt used a mistyped source SHA (`f0609445…` vs the actual `f06094431…`). OOD=0 completed but failed provenance audit; OOD=0.5 was stopped at 500 scan rows before scoring. Both attempts are preserved in `evidence/qcgs-exact-objective-audit/decisions/profile-source-revision-mismatch-20261008.json` and excluded from profile conclusions.
- Corrected profile-only jobs are running serially on the same excluded IDs: `profile-ood-0-s0-v12r1-20261008` (config SHA `68ae21e62ea799269e903a89e5d89014e261134f291ab08c0179710a88f2ac45`) and `profile-ood-05-s0-v12r1-20261008` (config SHA `643fda47be89d1586b18242b1fc1e7b5705dd4c63e12b4893f66c36047bdedbc`). Logs: `log/qcgs-exact-objective-audit/profile-only-v12-correct/`. Stage A has not launched.

## 2. Phạm vi và nguyên tắc thực thi

- Bắt đầu với CIFAR-100-C, CLIP ViT-B/16, OOD=0 và OOD=0.5, fixed Ramen candidate space, exhaustive legal one-swap actions và no-swap.
- Giữ baseline parity: cache admission/history, batch visibility, retrieval, support cardinality, class balance, weighting, optimizer, parameter reset và numerical ordering. Thay candidate pool hay update rule là một experiment khác.
- Nhãn thật, ID/OOD metadata, corruption/domain thật, supervised query gradients và oracle utility thuộc evaluator. Observation/scoring code không nhận các trường này hoặc đại lượng suy ra từ chúng.
- Objective definitions, sign conventions, controls, registry, coverage rules, cost budget và numerical GO/STOP gates phải được khóa trước scored execution.
- Không tune estimator, threshold, số views, candidate M/k hoặc learned scorer trước khi exact objective được xác nhận có thông tin. Không mở Stage B trước information gate.
- Giữ nguyên historical protocols/gates và negative runs. Sửa sai bằng revision/decision record mới; không ghi đè dự đoán hoặc kết quả cũ.
- Phân biệt CE utility, accuracy, ranking correlation và causal claims. OOD cells khác query/cache history không tự tạo ra causal OOD contrast.
- Toàn bộ chu kỳ có thể kết thúc bằng GO, CONTINUE, PIVOT, STOP hoặc INCONCLUSIVE có lý do. Failed execution không phải negative scientific result.

## 3. Tổ chức artifact và trách nhiệm

Tái sử dụng cấu trúc repo; chỉ tạo artifact khi task tương ứng bắt đầu:

| Artifact | Vị trí dự kiến | Owner |
|---|---|---|
| Research state, case/experiment cards, ledger, decisions, reports | Thư mục plan này | Integrator; worker được giao file cụ thể |
| Literature verification và problem report | `reports/literature-verification.md`, `reports/problem-map.md` trong thư mục plan | Literature worker |
| Protocol khoa học | `docs/research/qcgs-exact-label-free-objective-audit.md` | Integrator |
| Config và registry contract | `cfg/research/qcgs-exact-objective-audit/` nếu implementation cần | Coding worker |
| Runner/analysis/regressions | Module hiện có phù hợp trong `src/`, `tests/`, notebook nếu cần | Coding worker |
| Raw evidence, locks, receipts, logs, worker packets | `evidence/qcgs-exact-objective-audit/<run-id>/`, `log/qcgs-exact-objective-audit/` | Launcher |
| Source/config/registry hashes và summary kiểm chứng | Reports trong thư mục plan, liên kết tới raw evidence | Integrator |
| Claim matrix và evidence packet | `reports/claim-evidence-matrix.md`, `reports/evidence-packet.md` | Integrator |

Raw evidence/logs lớn giữ Git-ignored. Reports phải cho biết cách khôi phục artifact và hash khi raw không có trong checkout. Không sửa các file của skill để lưu project state.

IDs dùng trong chu kỳ: `CASE-QCGS-01` (exact objective audit), `CASE-QCGS-02` (reliability/enrichment), `CASE-QCGS-03` (estimator/confirmation, có điều kiện); experiment IDs `EXP-QCGS-001...`; claim IDs `CLAIM-QCGS-001...`. Ghi liên kết giữa card, command, run, analysis và decision.

### Worker và Git

- Launcher dùng [CODEX-WORKER-OPS.md](../../docs/CODEX-WORKER-OPS.md); không đưa file launcher đó hoặc `orchestrator.md` vào task packet của worker.
- Coding worker: fresh context `gpt-6-luna`, reasoning `max`. Reviewer code: fresh context `gpt-6.1-sol`, reasoning `low`. Không tự thay model/effort; nếu launcher không hỗ trợ cấu hình đã chỉ định thì báo chính xác phần không khả dụng.
- Ưu tiên worker tools của session khi đặt được đúng cấu hình. Khi dùng Codex CLI, kiểm tra options đang hỗ trợ trước; inherit sandbox/approval constraints, đóng stdin và dùng background execution có handle theo dõi. Không dựa vào slot của built-in agents để giới hạn các CLI worker độc lập.
- Packet có một objective, verified inputs, permitted edits, ownership, dependencies, budget, acceptance criteria và kết quả cần trả. Không chuyển cả lịch sử hội thoại.
- Reviewer chỉ tìm lỗi correctness, metric, leakage, exhaustive coverage, reset/parity, provenance hoặc vi phạm protocol trong diff. Không nhận đề xuất kiến trúc/refactor/benchmark vượt câu hỏi này nếu không có lỗi cụ thể cần giải quyết. Integrator ghi disposition cho từng finding.
- Trước worktree phục vụ code, commit đầu vào/protocol trên branch mới và xác nhận identity qua Git config của người dùng. Tạo worktree riêng cho các writer độc lập, ghi base commit và file ownership; reviewer đọc diff sau implementation. Không cho hai writer sửa cùng artifact.
- Integrator sở hữu quyết định khoa học và integration. GPU campaign do launcher điều phối; không để các worker tự launch inference cạnh tranh cùng device.

## 4. Dependency và các wave song song

```text
P0: reconstruct evidence/state + kiểm tra khả dụng data/runtime
  ├── P1: verify literature/problem map
  └── P2a: inspect instrument/parity/candidate/registry reuse
         ↓ reconcile P0 + P1 + P2a
P2: revised Question + hypotheses + freeze audit protocol
  ├── P3a: implementation exact audit
  ├── P3b: registry scan/overlap checks theo sampling rule đã khóa
  └── P3c: chuẩn bị cards, provenance, known-case expectations
         ↓ independent code review + checks + freeze executable source
P4: L0 exhaustive audit → L1 information screen
         ↓ validated decision
P5: L2 reliability / L3 enrichment, có điều kiện
         ↓ exact information gate đạt
P6: L4 estimator → L5 frozen confirmation, có điều kiện
         ↓ hoặc L6 STOP/PIVOT khi evidence không hỗ trợ ranking
P7: case review + claim matrix + evidence packet
```

Wave 1 song song literature verification với read-only instrument/data audit; không có implementation scoring trong wave này. Wave 2 chỉ bắt đầu sau khi protocol freeze: code, registry và provenance preparation tách ownership. Review phụ thuộc code đã có diff; analysis chính thức phụ thuộc raw đã hoàn tất. GPU jobs chỉ chạy song song nếu device/memory budget và baseline parity cho phép, có owner và log riêng.

## 5. P0 — Reconstruct research state và fatal-risk probes

Module: Establish context, Instrument. Budget: đọc reports/locks/raw khả dụng và preflight nhẹ; không rerun historical CUDA chỉ để đồng bộ roadmap.

- [x] Ghi branch/commit, working-tree state, runtime, model/data access, GPU availability và evidence inventory trong research state.
- [x] Đối chiếu [oracle Pilot B](../20260907-oracle-support-utility/reports/pilot-b-completed.md), [single-view results](../20260919-qcgs-diagnostic/reports/results.md), [multi-view results](../20260919-qcgs-multiview-rescue/reports/results.md) và [scientific-validation receipt](../20260919-qcgs-multiview-rescue/reports/scientific-validation.json).
- [x] Phân biệt screened oracle với exhaustive oracle; Stage A với Stage B; CE gains với prediction transitions. Ghi sample overlap, cells, controls và giới hạn offline validation.
- [x] Kiểm tra raw referenced artifacts; raw archive vắng nên metrics cũ không được recompute/reproduce trong chu kỳ này.
- [x] Kiểm tra split/label semantics, official dataset/checkpoint, locked stream reconstruction, cache visibility và reset. Xác định runtime/GPU/candidate-action memory là rủi ro trước audit.
- [x] Kiểm tra runtime theo [experiment-runtime.md](../../docs/research/experiment-runtime.md): local uv Python 3.11/PyTorch 2.6.0/CUDA 12.4 khác historical PyTorch 2.4.1; khác biệt ghi ở research state.
- [x] Fatal checks: dữ liệu/checkpoint qua official hashes/preflight; không có evidence mới để phát hiện lỗi reset/parity, nên CUDA smoke vẫn là gate bắt buộc. Thiếu raw cũ và extra README ở HF helper được ghi riêng, không coi là dữ liệu hỏng.

**Exit:** có state/evidence inventory phân biệt observed, reported và missing; một uncertainty ưu tiên: exact label-free outcome có order supervised utility hay không. Chưa khẳng định deployable selector hoặc impossibility.

## 6. P1 — Literature verification và problem map

Module: [Literature](../../.agents/skills/research-orchestrator/references/map-research-literature.md). Chạy song song P2a sau inventory tối thiểu P0.

- [x] Verify metadata/central relevance của toàn bộ 18 representative references và uncited DeYO, Tip-Adapter/TDA, TracIn, GRAD-MATCH, GLISTER, Test-Time Training; phân biệt publication status trong [literature verification](reports/literature-verification.md).
- [x] Kiểm tra nguồn proceedings, primary PDFs/HTML và code pages; đọc method/results ở action-identifiability, objective, retrieval, selective adaptation và calibration papers dùng để đặt câu hỏi/gates. Scope từng nguồn được ghi trong note; không coi abstract-only checks là reproduction.
- [x] Ghi source, status, mechanism, evidence boundary và unresolved claims. Các 2026 preprints/workshops/journal records được đánh nhãn rõ; source inaccessible không dùng làm central evidence.
- [x] Map 10 mechanism families synthesis §7–20 trong problem map, bao gồm entropy, selective/reliable TTA, open-set, multi-view, CLIP objectives, cache, influence/data valuation, unlabeled selection, identifiability, decision theory.
- [x] Targeted search local support action/data utility, method selection, risk estimation, selective adaptation and consensus/calibration; queries/date/coverage/limits ở [literature verification](reports/literature-verification.md).
- [x] Re-rank G1–G5 qualitatively: G1 broad action identifiability and G3 selective adaptation are prior work; only local support-action objective-to-utility audit stays open; G2 is represented by a frozen source-aligned objective; G4/G5 conditional.
- [x] Narrow claims and distinguish objective alignment from estimator quality; no entropy/margin, view-variance/bias, or local empirical STOP/general impossibility conflation.
- [x] Broad search exits at a decision-relevant stopping point, not a novelty proof; follow-up citation/code review remains a paper-facing limitation.

**Exit:** source-backed problem map, metadata/claim verification status, candidate gap và limits; novelty vẫn provisional nếu current/adjacent search thiếu. Budget review theo việc nguồn mới có đổi quyết định hay không, không theo số paper tùy ý.

## 7. P2 — Revised Question và freeze protocol

Module: [Question](../../.agents/skills/research-orchestrator/references/frame-research-question.md). Đầu vào P0/P1 và P2a read-only audit. Đây là bước immediate action của synthesis §39; hoàn tất trước viết runner mới hoặc mở campaign.

### Hypothesis tree và đối chứng

| Hypothesis | Signature cần đo | Alternative / falsifier |
|---|---|---|
| H1: objective mismatch | Exact entropy improvement vẫn không order utility ổn định | Exact entropy có thông tin nhưng first-order estimate sai: mở estimator question thay vì đóng objective |
| H2: VLM-aligned evidence | Source/text/contrastive channel có alignment/top-action utility tốt hơn entropy trong cả cells | Confidence scaling hoặc extra compute giải thích gain; dùng matched observation/control |
| H3: reliability precedes ranking | CAS may predict when a fixed Ramen action/policy should be skipped; open only if Stage A leaves a nonconstant useful target. | Selective adapt/skip is prior work; CAS transfer from episodic TPT may fail in online Ramen. Thresholds fitted on utility labels are supervised development. |
| H4: bounded evidence insufficiency | The three exact objective formulations may fail to order utility in this frozen Ramen state. | Instrument defect, imprecision or an unmeasured channel; finite failures do not prove general non-identifiability. |

### Frozen objective contract and deferred families (§25)

Mỗi candidate phải có source-backed information hypothesis, formula, observation access, anchor state, direction, required forwards/gradients, validity rule và cost. Chọn bộ đại diện nhỏ nhất để phân biệt mechanisms, ghi lý do deferred variants; không biến danh sách thành sweep mọi heuristic.

| Family | Frozen candidate | Control |
|---|---|---|
| Entropy | Exact post-action Shannon entropy on the original query view | Fixed prompts/temperature; lower loss defines positive direction |
| Multi-view | MEMO marginal entropy over eight fixed `four-corners-30-flip-v1` views | Same deterministic views for all actions; no label/target selection |
| Source-aligned | Query-row image→text entropy component of CLIPTTA batch soft-contrastive loss | Frozen pseudo-caption bank from Ramen θ_R; no OCE, memory or extra terms; call it CLIPTTA-derived, not a reproduction |
| Reliability/counterfactual/neighborhood | CAS, DeYO response, cache consensus and other channels | Deferred; CAS is established selective-adaptation work. Open only with a new case card and fresh confirmation. |

Observation đòi hỏi thêm views/context/teacher được budget và đánh dấu evidence enrichment; thu thập theo các block riêng để so với local channel. L3 chỉ đánh giá thêm family khi local evidence thất bại; không gộp các channel trước khi biết thông tin mỗi channel đóng góp.

### Sampling và exact-action contract

- [x] Candidate construction/top-k/top-M, canonical order, legal same-pseudo-class one-swap and no-swap frozen in [exact objective protocol](../../docs/research/qcgs-exact-label-free-objective-audit.md).
- [x] Every action resets to θ₀ and applies the exact one-step Ramen update against one cache snapshot; trial state never carries forward.
- [x] Evaluator-only utility, objective signs, no-swap values, constant-rank handling and canonical ties frozen in the protocol.
- [x] Random draw/expectation, same-anchor supervised first-order reference and exact Stage-A oracle specified separately.
- [x] Fresh outcome-blind registry uses 1,499 historical query exclusions plus smoke IDs and a fixed modulo-9 Stage-A/B split; prior unscored cache-stream exposure remains a stated limit.
- [x] Split/ID metadata stays in registry/evaluator. Correctness, headroom, objective score and CE utility never select queries.
- [x] Seed, severity, split, stream, quotas, block size, scan cap, shortage and resume rules frozen.
- [x] Stage B is disjoint and reserved; it opens only after the objective gate or a separately documented P5 case.

### Những trường bắt buộc để protocol được gọi là frozen

Scientific gates và compute budgets theo linked protocol; memory launch policy áp dụng revision hiện hành ở trên. Cập nhật protocol/config/validator cùng hashes trước scored run; không dùng cap 7,5 GiB cũ để chặn registry. Integrator vẫn freeze executable details:

- Exact candidate definitions, observation states, sign/tie/no-swap rules và family/block order.
- Minimum eligible queries/cell, action/objective coverage, minimum valid-rho fraction, handling constant vectors/undefined values và target precision.
- Numerical meaningful-effect/ranking/sign-precision/top-utility criteria; uncertainty estimator/resampling unit; cross-cell stability và outlier sensitivity limits; multiplicity/selection rule giữa objectives.
- Reliability branch remains deferred; CAS evaluation requires a new P5 card/addendum and fresh use of reserved queries.
- Source/config/registry/exclusion versions, runtime/device, action chunk size, actual smoke cost and launch caps already specified in protocol.
- Exact launch command, output schema, campaign run IDs, validation checklist and review date are frozen in the P3 launch lock after the runner CLI exists.

Giá trị unresolved chặn scored campaign, không chặn literature/instrument work. Cost-only smoke dùng registry riêng, không dùng để chọn objective theo outcome; results exploratory và exclusions được lưu.

**Exit:** `CASE-QCGS-01` and scientific protocol freeze are complete. Commit before creating the code worktree. P3 adds executable command, source/config/registry hashes, smoke-safe action chunk and launch lock before any scored query is evaluated.

## 8. P3 — Instrument implementation, review và reproducibility gate

Module: [Instrument](../../.agents/skills/research-orchestrator/references/build-research-instrument.md). Coding worker triển khai contract P2 bằng cách tái sử dụng instrument hiện có sau audit; không viết deployable selector ở L0.

**Trạng thái:** reviewed source fix `04abdba`, CPU validation và hai CUDA smoke complete ở coding worktree. Còn reconcile vào execution checkout, áp dụng resource-policy revision và hoàn tất registry/launch lock. Không mở thêm memory-feasibility case chỉ vì peak vượt 7,5 GiB.

### Findings cần disposition trước scored launch

| Finding từ source review | Disposition và evidence còn lại |
|---|---|
| Grouped isolation check chỉ lấy mẫu hai query đầu của action đầu. | Đã ghi `action_chunk_isolation_scope` và số forward checks trong raw receipt. CPU regression so sánh kết quả chunked/serial cho toàn bộ action/query fixture. Launch config mặc định chunk 1. CUDA receipt vẫn phải báo số check thực và không gọi sampled check là exhaustive. |
| Raw audit không giữ tất cả objective tensors để tính lại từng raw value. | Đã thêm independent direct-formula replay trên tensor logits/view logits/feature/caption bank ở mọi baseline/action scorer call trong deterministic fixture; mapping caption bank so với per-row Ramen predictions được assert. Raw campaign analyzer vẫn xác minh scalar arithmetic, hashes và source/config/protocol provenance; report nêu rõ nó không dựng lại mọi tensor objective từ JSONL. |
| Cost trên CLIP CUDA/8 GiB. | Đã đo hai smoke complete trong 35:08, peak 7,785 / 7,794 GiB. Peak là reported cost; mức dự phòng 7,5 GiB không còn là blocker. Báo runtime/action counts và extrapolation limits cho Stage A. |
| Stage B tính ba objective values dù chỉ lưu winner. | Chưa tối ưu trước `GO_CONFIRM`; đo cost smoke/Stage A và chỉ xem lại nếu đây là cost blocker. Bất kỳ thay đổi nào phải giữ winner score/selection và observation access theo protocol. |

### Thứ tự hoàn tất P3

1. Reconcile reviewed source `04abdba` cùng findings/dispositions và receipts từ coding worktree; giữ label isolation, baseline/reset/cache và exhaustive coverage.
2. Reuse required review/regressions/suite đã hoàn tất; chỉ kiểm phần validator/provenance bị ảnh hưởng bởi revision. Freeze source/protocol/config hashes dùng cho scored run, không mở lại vòng full validation khi không có thay đổi liên quan.
3. Reuse hai unscored scans và hai CUDA smoke complete trên source fix `04abdba`; kiểm receipts về finite objectives, parity/reset/cache, coverage, mapping và cost. Cập nhật resource clause/validator theo revision trước scored run, ghi hash mới và provenance reuse. Không dùng smoke utility để tune objective, không yêu cầu rerun chỉ vì peak >7,5 GiB.
4. Profile bottleneck trên cost-only smoke workload và tối ưu cục bộ để giảm thời gian experiment theo mục trên; kiểm before/after và parity của execution path thay đổi. Chốt chunk size và dự toán Stage A (≤8 GPU-hours/cell, ≤16 tổng). Peak memory là measurement; không giảm actions/views/quota hay đổi scientific gates, không mở case chỉ để đạt mức dự phòng.
5. Freeze registry, exact CLI, source/config/protocol/transform hashes, smoke receipt và unique run IDs trong launch lock; cập nhật live state/readiness. Chỉ sau đó mở scored Stage A.

- [x] Inspect `src/evaluation/oracle_support_utility.py`, `src/runtime/query_gradient_registry.py`, existing diagnostic/notebook paths và relevant configs; xác định phần reuse và sửa tối thiểu cần để đo exact objectives.
- [x] Ghi action records đủ tái tính: query/action opaque keys, candidate snapshot/hash, baseline/action outcomes cho objective, evaluator CE/predictions riêng, valid/undefined reasons và costs.
- [x] Validate formula/sign trên known cases: no-swap=0, reward/loss orientation, rank ties, constant objective, empty positive set, action ordering, random expectation và non-finite/undefined treatment. Direct scorer replay test checks all three formulas and caption-bank mapping; per-run tensor replay remains outside the raw JSONL analyzer.
- [x] Regression cho label isolation, trial order independence/reset, unchanged cache, baseline exact parity và exhaustive action coverage. Focused suite 32 pass; full suite 440 pass (1 skipped). CPU checks không chứng minh CUDA parity.
- [x] Reviewer fresh context kiểm diff theo protocol và acceptance criteria; fix concrete findings rồi rerun checks bị ảnh hưởng. Final targeted `gpt-6.1-sol`/low review reported no remaining findings in the last diff.
- [x] Freeze implementation source tree, config bundle, protocol SHA-256 và commit reviewed source trước CUDA; commit `b33549ad2e4e35e349a42909f7dbd4e32286ce58` dùng Git global identity đã khóa.
- [ ] Freeze registry/exclusion, smoke receipts, exact commands/run IDs và hoàn tất launch lock sau scans + smoke, trước scored Stage A; ghi runtime/device differences và reproduction commands.
- [x] Hai real CUDA smoke trên separate queries đã complete trên source `04abdba`; raw receipts ở `evidence/qcgs-exact-objective-audit/`. Reconcile receipts vào launch lock theo resource-policy revision; utility smoke không dùng để tune objective.

Các command đã có trong repo, thực thi sau khi xác nhận data root/runtime phù hợp:

```sh
uv sync --locked --offline
source .venv/bin/activate
PYTHONPATH=src python -m runtime.preflight --data-root ~/data --dataset CIFAR100C
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

`--offline` chỉ dùng khi dependencies đã cached; thiếu cache phải được ghi nhận và xử lý trong runtime setup. Không để tests tự tải data/checkpoint hoặc launch campaign. Campaign CLI/arguments sẽ ghi đúng theo implementation, không dùng placeholder command như thể đã chạy.

**Exit:** findings có disposition, required checks và CUDA smoke có receipts; parity/reset/access/exhaustiveness được chứng minh trong phạm vi đã ghi; revised resource policy và registry/launch lock hoàn tất. Peak >7,5 GiB không chặn exit. Campaign cần audit raw riêng.

## 9. P4 — L0 exhaustive execution và L1 information screen

Module: [Research loop](../../.agents/skills/research-orchestrator/references/run-research-loop.md). Budget đúng launch plan P2/P3; lưu actual costs và failed attempts.

**Bước kế tiếp sau P3:** chỉ launch Stage A 16 registered queries/cell và ba objectives đã freeze. `GO_CONFIRM` chọn một winner theo protocol trước khi mở Stage B 128 fresh queries/cell. `GO_P6` chỉ có sau independent Stage-B confirmation; Stage-A screening không trực tiếp mở estimator design.

- [ ] Launch complete preregistered batches cho hai OOD cells bằng unique run IDs. Theo dõi process/job handles, attempted/completed counts, checkpoints và bounded waits; không restart từ timeout quan sát nếu handle vẫn sống.
- [ ] Stage A đo exact objective và evaluator utility của mọi legal action từ cùng snapshot. Stage B chỉ score frozen winner trên mọi actions; evaluator chỉ đo selected action, no-swap và một random-forced draw, suy ra random-matched từ cùng draw/abstention. Lưu records theo đúng stage contract; bảo toàn negative/harmful examples.
- [ ] Verify provenance, registry membership/overlap, query quotas, action enumeration độc lập, finite utility và objective validity; recompute metrics từ raw thay vì tin summary runner.
- [ ] Stage A ranking: Spearman per query (tied ranks), mean/median, positive-rho fraction, valid/total query count; Kendall chỉ thêm khi trả lời câu hỏi mới. Không tính/release action-level ranking ở Stage B.
- [ ] Stage A sign prediction: `P(U>0 | delta_z>0)`, số positive actions/queries, helpful/harmful counts; báo cả query-balanced và pooled nếu dùng, không coi actions cùng query là independent samples. Stage B chỉ báo selected-action harm/helpfulness.
- [ ] Top action: argmax objective theo canonical ties/no-swap rule đã khóa; utility so no-swap và random controls. Supervised first-order, exact oracle, expected random và regret `U_oracle-U_selected` chỉ thuộc Stage A.
- [ ] Stability: hai OOD cells, corruption groups, baseline-correct/wrong, uncertainty strata và memory states. Correctness/OOD strata chỉ dùng cho evaluator diagnosis, không làm reliability gate inputs.
- [ ] Sensitivity: mean/median/trimmed mean, harmful/helpful magnitude, largest-contributor removal rule đã freeze, harm rate và uncertainty theo query hoặc stream blocks phù hợp dependency.
- [ ] Báo prediction transitions, entropy/confidence, calibration (binning/estimator đã khóa khi dùng), ID/OOD effects và cost. Không báo true-label CE cho OOD dưới nhãn known-class giả.
- [ ] Review mỗi objective/family theo gate dưới; giữ negative và undefined coverage trong reports. Không lấy pooled positive result để che cell đảo chiều.

### Decision order dùng cho mọi block

| Decision | Evidence bắt buộc / hành động |
|---|---|
| INVALID | Provenance/leakage/parity/candidate mismatch, non-finite values trái contract hoặc campaign declared complete nhưng bỏ legal actions: repair instrument, giữ invalid run, không scientific STOP |
| INCONCLUSIVE | Quota thiếu, actual execution/resource interruption, compute budget hết hoặc <4 scored blocks/cell: ghi thiếu gì; extension theo plan/revision riêng. Peak memory cao với run complete không tự tạo INCONCLUSIVE. Constant objective hoặc <8/16 defined rho do declared valid reasons là objective gate failure. Descriptive intervals không tạo thêm gate ngoài protocol. |
| STOP OBJECTIVE FAMILY | Exact objectives đã kiểm trong family không có positive ranking/selected utility ổn định so controls hoặc cross-cell reversal theo numerical gates: đóng các formulations/access đã kiểm, không claim mọi entropy/label-free objective bất khả thi |
| GO_CONFIRM | Complete valid Stage A có ít nhất một objective qua tất cả frozen §7 investment gates ở cả cells: freeze một winner theo mean rho/tie rule và mở fresh Stage B; chưa mở estimator. Sign precision và exploratory strata được báo cáo, không tự thêm thành gates. |
| GO_P6 | Complete valid Stage B qua frozen utility/random-matched/Holm, median, contributor-removal, coverage và transition gates: mở case estimator riêng cho winner; chưa gọi là deployable success. |
| CONTINUE RELIABILITY/ENRICHMENT | Universal ranking thất bại nhưng câu hỏi H3/H4 còn informative và có block budget: mở L2/L3 theo card mới, không tune lại objective bị STOP |

Expected missing records trong interrupted campaign là incomplete/inconclusive; lỗi enumeration của instrument là invalid. Undefined objectives có declared reason/coverage limit khác với silent NaN hay utility không hữu hạn.

**Exit:** raw audit + objective/family decision records, hypothesis interpretation và next question. Positive top-action CE không tự chứng minh reliable ordering hay accuracy gain.

## 10. P5 — L2 reliability, L3 evidence enrichment và L6 STOP

Mở `CASE-QCGS-02` chỉ khi P4 decision chỉ ra giá trị thông tin còn lại, với budget và pre-run card mới.

- [ ] L2 phân biệt target `max_a U(q,a)>0` với `U(q,a_selected)>0`; oracle headroom widespread có thể khiến target thứ nhất gần constant. Kiểm tra trivial always-adapt/always-abstain và base rates trước learned gate.
- [ ] Observable reliability variables có thể dùng uncertainty/view response/source consistency/memory consensus đã định nghĩa; không dùng baseline true correctness, true OOD hoặc oracle quantities.
- [ ] Nếu fit gate/threshold, giữ development và independent evaluation tách biệt; báo label access trong gate training. Gate học từ utility labels là supervised development, không được gọi là hoàn toàn unsupervised chỉ vì inference label-free.
- [ ] Báo risk/utility–coverage, no-swap baseline, conditional utility, harm, transitions và uncertainty; không chọn operating point bằng confirmation outcomes.
- [ ] L3 thêm từng conceptually distinct channel theo marginal-information question: source-anchor, multi-view, counterfactual, neighborhood/temporal. Giữ candidate/update fixed, matched compute/access controls và measurement block riêng.
- [ ] Không mở heuristic mixture/learned ranker để cứu failed exact signals khi chưa có thông tin của individual channel. Reuse raw được phép khi objective đã đo và analysis exploratory được ghi rõ.
- [ ] L6 định nghĩa distance/tolerance trên accessible evidence và comparable action representations trước collision analysis; báo pairs/support, reversed orderings, sensitivity và missing channels.
- [ ] Chỉ kết luận limitation trong measured observation channel/setting. Finite negative audit hoặc evidence-near collision không chứng minh mathematical non-identifiability nếu không có assumptions/theorem phù hợp.

**Exit:** ranking GO cho một channel, selective adaptation question có evidence đủ mở confirmation, hoặc STOP/PIVOT sang population selection/bounded guarantees/failure characterization. Mỗi extension phải có uncertainty mới và stop signal; không lặp lại entropy-scoring engineering.

## 11. P6 — L4 estimator và L5 frozen confirmation (có điều kiện)

Mở `CASE-QCGS-03` cho winner của `CASE-QCGS-01` chỉ sau validated `GO_P6` từ Stage B. Channel mới ở P5 cần information/independent confirmation gates của case riêng trước estimator. Nếu upstream gate không đạt, phase này ghi `not reached` với upstream decision; không gọi là implementation thiếu.

- [ ] Tách fidelity với exact objective (`estimate_delta_z` so `delta_z`) khỏi alignment với supervised utility. Chọn estimator nhỏ nhất phù hợp geometry/update: gradient dot/sign-aware, finite difference/JVP; HVP/learned ranker chỉ khi failure signature và cost chứng minh cần thiết.
- [ ] Coding/review theo cùng model/effort và scope P3. Giữ label isolation, reset/cache/baseline parity; compare exact-objective upper bound và matched compute controls.
- [ ] Freeze objective, observation access, estimator, parameters, threshold, tie/no-swap/abstention rule, cost cap, source/config, Stage B fresh registry và GO/STOP gates trước confirmation.
- [ ] Chạy frozen confirmation độc lập; record tất cả attempts và deviations. Confirmation dùng để ra quyết định, không tiếp tục tune ngay trên cùng queries.
- [ ] Kiểm utility distribution, transitions/accuracy, ranking/regret, coverage/calibration, open-set memory effects và deployable latency/memory/gradient counts. Exhaustive evaluator cost không được báo thay deployable latency.
- [ ] Broader dataset/corruption/seed tests chỉ theo generalization claim cần kiểm; đưa mỗi expansion vào case card có budget/exit. Không tự mở full matrix chỉ vì một cell dương.

**Exit:** validated confirmation hỗ trợ claim bounded, hoặc STOP/PIVOT có failure analysis. Không có requirement rằng QCGS-v3 hay phương pháp mới bắt buộc được tạo.

## 12. P7 — Case review, claim evidence và handoff

Modules: [Case review](../../.agents/skills/research-orchestrator/references/review-research-case.md), [Paper evidence](../../.agents/skills/research-orchestrator/references/build-paper-evidence.md). Thực hiện sau mỗi meaningful cycle và tổng hợp khi nhánh đạt exit signal.

- [ ] So kết quả với pre-run hypotheses/gates, audit validity trước interpretation; ghi returns về capability, infrastructure và scientific insight.
- [ ] Decision record chọn continue/advance/pivot/stop, evidence thuận/nghịch, strongest alternative, unresolved risks, budget đã dùng và next action với success/stop signals.
- [ ] Update live state và roadmap bằng liên kết decisions; không ghi đè lịch sử STOP hay biến reported evidence thành mới reproduced.
- [ ] Claim matrix phân biệt oracle opportunity, exact-objective information, estimator fidelity, selective/deployable benefit và identifiability limitation; mỗi claim có supporting raw/report, uncertainty, contradiction và permissible wording.
- [ ] Figures/tables cho objective–utility rankings, family/cell comparisons, harm/transition distributions, reliability coverage, estimator fidelity và cost nếu nhánh tương ứng thực sự có evidence.
- [ ] Evidence packet gồm question, reference/change, decisive valid evidence, claim limits, source search/verification coverage, reproducibility commands/versions/hashes và next decision.
- [ ] Outline hoặc grounded research note/paper sections theo evidence; negative audit có thể là kết quả khoa học. Không tự khẳng định novelty, universal impossibility, causal mechanism hay generalization chưa đo. Publishing/submission không thuộc plan này.

## 13. Coverage toàn bộ synthesis và completion audit

| Phần synthesis | Task giải quyết | Bằng chứng hoàn tất cần kiểm |
|---|---|---|
| §1–5: synthesis, inventory, oracle và hai STOP | P0 | Inventory + report/raw reconciliation; missing/offline limits rõ |
| §6–20: failure map, 10 literature families, theory | P1, P2 | Verified source notes + problem map + alternative explanations |
| §21–23: G1–G5, revised Question, H1–H4 | P2 | Operational hypotheses, discriminating signatures, case cards |
| §24–26: exact audit, 5 objective families, metrics | P2–P4 | Frozen objective contracts, action records, raw-derived metrics |
| §27: L0–L6 staircase | P4–P6 | Decision cho mỗi reached layer; not-reached layers có upstream gate reason |
| §28–30: engineering prohibition, novelty, abstention | P1, P2, P5, P6 | Không premature estimator; scoped novelty/search và reliability evidence |
| §31: evaluation dimensions | P4, P5, P6 | Utility, transitions, ranking, reliability, calibration, OOD, cost hoặc justified N/A |
| §32–35: saturation, transition, next scope | P1, P2 | Saturation rationale + state transition + fixed initial setting |
| §36: GO/STOP logic | P2, P4–P6 | Numerical gates freeze trước run + independent audit/decision |
| §37–39: interpretation, references, immediate action | P1, P2, P7 | Reference verification, 7-item protocol, evidence-grounded handoff |

Trước khi đánh dấu toàn bộ chu kỳ hoàn tất:

- [ ] Audit từng requirement trên và mọi explicit artifact/command/gate của protocol đã frozen; đọc authoritative files/raw/process status, không dựa vào intent hay roadmap checkbox.
- [ ] Mọi required run đã terminal và có validity disposition; counts attempted/completed/validated/missing nhất quán. Mỗi command được ghi đúng là ran/planned/failed.
- [ ] Các layer sau chỉ được đóng theo gate có evidence; `not reached` hợp lệ do scientific STOP/PIVOT, không dùng để che unexecuted required work hoặc thiếu data/compute.
- [ ] Tất cả claims khớp verification scope. Missing raw, inaccessible sources, inconclusive precision hay unverified result còn là open work trong scope đã chọn.
- [ ] Durable reports/decisions/evidence packet được lưu; source/config/registry/protocol hashes và artifact recovery đủ để người khác kiểm lại.
- [ ] Handoff nêu câu hỏi đã giảm uncertainty gì, còn alternative nào, quyết định hiện tại và next bounded test. Không cần mọi hypothesis đúng để nghiên cứu hoàn tất; cần decision được chứng minh và không còn required task chưa xử lý.

## 14. Việc thực hiện đầu tiên khi chạy plan

1. Reconcile reviewed source `04abdba` và completed smoke receipts từ coding worktree vào execution checkout; cập nhật live state và giữ historical attempts.
2. Áp dụng revision bỏ cap dự phòng 7,5 GiB trong protocol/config/validator; profile runtime bottleneck và tối ưu cục bộ trên cost-only smoke IDs, kiểm speedup/parity trên cùng workload. Ghi source/config/protocol hashes mới và validation phù hợp cho code thay đổi.
3. Freeze scientific registry/launch lock tối thiểu và chạy Stage A 16 queries/cell trong compute budget; theo dõi actual memory/runtime/OOM, audit raw và ghi scientific decision. Không tiếp tục tối ưu phần không phải bottleneck để trì hoãn phép thử.
4. Chỉ GO_CONFIRM mở Stage B; chỉ GO_P6 mở estimator case. Actual execution failure được sửa hoặc chuyển hardware với failed attempts được giữ; P5 extension cần câu hỏi mới, case card và budget.

Việc viết plan này không tự launch workers, tests hoặc CUDA campaign. Khi thực thi, launcher tiếp tục theo authorization hiện có, không chèn approval gates vào từng variant; chỉ dừng để lấy thông tin thực sự thiếu khiến bước phụ thuộc không thể tiến hành đúng.
