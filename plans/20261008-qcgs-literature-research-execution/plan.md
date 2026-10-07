---
title: "QCGS literature reopening → exact objective audit → research decision"
status: planned
branch: qcgs-objective-audit
created: 2026-10-08
owner: research integrator
scope: "Thực thi toàn bộ synthesis theo các gate nghiên cứu; các bước sau audit phụ thuộc evidence"
---

# Plan thực thi QCGS literature research synthesis

## 1. Mục tiêu và trạng thái

Đầu vào chính là [qcgs-literature-research-synthesis.md](../../docs/research/qcgs-literature-research-synthesis.md). Workflow theo [research-orchestrator](../../.agents/skills/research-orchestrator/SKILL.md): Literature → Question → Instrument → Research loop → Case review → Paper evidence; mở lại gate trước nếu evidence mới làm mất hiệu lực giả định.

**Câu hỏi trung tâm:** Trong candidate pool query-conditioned cố định của Ramen, observation channel không nhãn nào chứa thông tin để phân biệt support-gradient substitution có ích và có hại, và khi nào thông tin đó có thể chuyển thành ranking triển khai được?

**Đích khoa học:** Tách objective alignment khỏi estimator quality; đưa ra quyết định có bằng chứng về ranking, abstention hoặc giới hạn của observation channel. Một STOP hợp lệ có thể hoàn tất một nhánh nghiên cứu; không yêu cầu mọi nhánh tạo ra phương pháp tốt hơn.

Synthesis báo cáo oracle headroom, STOP của single-view entropy-sign và STOP Stage A của MV-target-θ_R. Plan coi đây là đầu vào cần đối chiếu với reports/raw artifacts, chưa mặc nhiên là kết quả đã được xác minh lại trong chu kỳ này. Roadmap cũ không quyết định trạng thái hiện tại.

**Deliverable của bước lập plan hiện tại:** file này. Các task dưới đây đều đang planned; chưa có protocol mới frozen, registry mới, implementation audit mới hoặc kết quả CUDA mới.

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

- [ ] Ghi branch/commit, working-tree state, runtime, model/data access, GPU availability và evidence inventory trong research state.
- [ ] Đối chiếu [oracle Pilot B](../20260907-oracle-support-utility/reports/pilot-b-completed.md), [single-view results](../20260919-qcgs-diagnostic/reports/results.md), [multi-view results](../20260919-qcgs-multiview-rescue/reports/results.md) và [scientific-validation receipt](../20260919-qcgs-multiview-rescue/reports/scientific-validation.json).
- [ ] Phân biệt screened oracle với exhaustive oracle; Stage A với Stage B; CE gains với prediction transitions. Ghi sample overlap, cells, controls và giới hạn offline validation.
- [ ] Kiểm tra raw referenced artifacts có thực sự tồn tại; recompute central metrics nếu raw đủ. Thiếu raw phải ghi unavailable, không gọi là reproduced/validated mới chỉ từ report hoặc receipt.
- [ ] Kiểm tra split/label semantics, official dataset/checkpoint, locked stream reconstruction, cache visibility và reset. Xác định tài nguyên nào cần chuẩn bị trước audit.
- [ ] Kiểm tra runtime theo [experiment-runtime.md](../../docs/research/experiment-runtime.md): local uv Python 3.11/PyTorch 2.6.0/CUDA 12.4 khác historical PyTorch 2.4.1; ghi khác biệt trước collecting evidence.
- [ ] Đánh dấu lỗi fatal: data không đúng, checkpoint thiếu, reset defect, label leakage, candidate-space mismatch hoặc baseline không tái dựng được. Ưu tiên instrument repair trước diễn giải scientific failure.

**Exit:** có state/evidence inventory phân biệt observed, reported và missing; một uncertainty ưu tiên: exact label-free outcome có order supervised utility hay không. Chưa khẳng định deployable selector hoặc impossibility.

## 6. P1 — Literature verification và problem map

Module: [Literature](../../.agents/skills/research-orchestrator/references/map-research-literature.md). Chạy song song P2a sau inventory tối thiểu P0.

- [ ] Verify metadata và central claims của toàn bộ 18 representative references ở §38; kiểm tra cả DeYO, Tip-Adapter/TDA, TracIn, GRAD-MATCH, GLISTER, Test-Time Training và các kết nối không có citation đầy đủ ở §11, §14–19.
- [ ] Đi từ link supplied/search tới original paper/proceedings/author code. Đọc method/experiments khi claim dùng để chọn objective hoặc đặt gate; abstract chỉ đủ để xác định relevance.
- [ ] Ghi title/authors/year/URL, access/date, sections đọc, author assertion, demonstrated evidence, inference của project và unresolved points. Reference 2026 hoặc theory chưa định danh phải verify riêng; inaccessible không được dùng như nguồn đã xác minh.
- [ ] Tổ chức source notes theo 10 mechanism families của §7–20: entropy, reliable/selective adaptation, open-set/noisy stream, augmentation/multi-view, VLM objectives, memory/cache, influence/data valuation, unlabeled risk/model selection, identifiability, decision theory.
- [ ] Tìm adjacent/current work theo support-action utility, unsupervised action/model selection, self-supervised gradient alignment và selective TTA; lưu ngày, queries, coverage và search limits.
- [ ] So sánh G1–G5 bằng importance, headroom, observability, feasibility và discriminating test. Chọn G1 làm ưu tiên nếu reconstructed evidence vẫn hỗ trợ; G2/G3/G4 là alternative mechanisms; G5 có dependency vào exact signal.
- [ ] Sửa hoặc hạn chế các claim trong problem report khi paper không hỗ trợ chúng. Không đồng nhất prediction margin với contrastive-pretraining objective, hoặc multi-view variance reduction với giảm bias.
- [ ] Kết thúc broad expansion khi targeted reading chỉ bổ sung vào mechanism classes hiện có và không đổi research question. Các central source chưa xác minh vẫn là task còn mở, không được che bằng tuyên bố saturation.

**Exit:** source-backed problem map, metadata/claim verification status, candidate gap và limits; novelty vẫn provisional nếu current/adjacent search thiếu. Budget review theo việc nguồn mới có đổi quyết định hay không, không theo số paper tùy ý.

## 7. P2 — Revised Question và freeze protocol

Module: [Question](../../.agents/skills/research-orchestrator/references/frame-research-question.md). Đầu vào P0/P1 và P2a read-only audit. Đây là bước immediate action của synthesis §39; hoàn tất trước viết runner mới hoặc mở campaign.

### Hypothesis tree và đối chứng

| Hypothesis | Signature cần đo | Alternative / falsifier |
|---|---|---|
| H1: objective mismatch | Exact entropy improvement vẫn không order utility ổn định | Exact entropy có thông tin nhưng first-order estimate sai: mở estimator question thay vì đóng objective |
| H2: VLM-aligned evidence | Source/text/contrastive channel có alignment/top-action utility tốt hơn entropy trong cả cells | Confidence scaling hoặc extra compute giải thích gain; dùng matched observation/control |
| H3: reliability precedes ranking | Observable gate chọn subset có utility tốt, abstention giảm harm, báo coverage | Gate dùng evaluator correctness/utility hoặc tune trên confirmation; kết quả không hợp lệ |
| H4: evidence insufficiency | Nhiều channel khác nhau thất bại; evidence-near states có reversed oracle ordering | Instrument sai, sample thiếu hoặc chưa đo channel quan trọng; empirical collision không chứng minh impossibility phổ quát |

### Objective contract theo năm family (§25)

Mỗi candidate phải có source-backed information hypothesis, formula, observation access, anchor state, direction, required forwards/gradients, validity rule và cost. Chọn bộ đại diện nhỏ nhất để phân biệt mechanisms, ghi lý do deferred variants; không biến danh sách thành sweep mọi heuristic.

| Family | Candidate cần đặc tả trước freeze | Kiểm soát quan trọng |
|---|---|---|
| Entropy/calibration | Shannon entropy, confidence/margin; calibrated/Tsallis variant khi nguồn và calibration access đủ rõ | Temperature/Tsallis parameters không fit bằng audit/confirmation labels; entropy giảm chưa đồng nghĩa CE tốt |
| Contrastive/source-anchor | Image-text prototype objective/margin, fixed-source agreement, soft contrastive objective hoặc frozen-teacher reward | Anchor/teacher/text prototypes frozen; phân biệt đúng formulation contrastive với confidence surrogate |
| Multi-view | Marginal entropy, view agreement, fixed-target CE, inlier-weighted consensus hoặc prediction variance | Fixed views/RNG; target construction không nhận label hoặc post-hoc utility; baseline forward budget được báo riêng |
| Counterfactual reliability | Response trước/sau update với perturbation có cấu trúc/object-destructive | Intervention và strength khóa; không coi object-destructive transform là semantic-preserving augmentation |
| Neighborhood/memory | Neighbor agreement, prototype/density consistency, support consensus hoặc temporal consistency | Chỉ memory visible tại đúng query state; không sửa cache qua action trials, không dùng future/evaluator metadata |

Observation đòi hỏi thêm views/context/teacher được budget và đánh dấu evidence enrichment; thu thập theo các block riêng để so với local channel. L3 chỉ đánh giá thêm family khi local evidence thất bại; không gộp các channel trước khi biết thông tin mỗi channel đóng góp.

### Sampling và exact-action contract

- [ ] Giữ candidate construction/top-k/top-M và canonical action order từ protocol baseline đã audit; legal action thay tối đa một support trong cùng pseudo-class, no-swap luôn hợp lệ. Không dùng supervised screening để giảm exhaustive action space.
- [ ] Mỗi action bắt đầu từ cùng pretrained/reset state và cùng cache snapshot, thực hiện exact Ramen support aggregate/update; không chạy trial này trên adapted parameters của trial trước.
- [ ] Utility evaluator-only: `U(q,a)=CE_y(theta_Ramen)-CE_y(theta_a)`, no-swap bằng 0. Với loss objective: `delta_z=z(theta_Ramen)-z(theta_a)`; với reward: ngược chiều. Không chọn chiều bằng observed correlations.
- [ ] Freeze random legal-swap control/RNG, supervised first-order evaluator reference và exact oracle. Oracle gồm no-swap; random-swap distribution và cách tính expected utility khóa riêng, không lẫn random swap với abstention.
- [ ] Fresh registry chọn bằng rule outcome-independent trước utility collection; exclude union query IDs của historical oracle/single-view/MV development và confirmation sets có thể truy xuất. Báo overlap theo source sample ID, corruption/view identity và stream exposure; thiếu exclusions không được tuyên bố fully fresh.
- [ ] ID eligibility/stratification cần evaluator metadata phải nằm ở registry/evaluator layer, không truyền sang scorer. Không chọn query theo Ramen correctness, positive oracle headroom, proxy success hoặc CE utility.
- [ ] Freeze seed, corruption/severity/split/stream, eligibility, quotas mỗi cell, scan caps và shortage policy. Scan thiếu quota chỉ tăng cap theo revision trước scored outcomes; không thay gate sau kết quả.
- [ ] Reserve confirmation registry disjoint; không dùng confirmation outcomes để chọn objective/threshold/variant. Development/exploratory và frozen confirmation có IDs/trạng thái riêng.

### Những trường bắt buộc để protocol được gọi là frozen

Plan không tự đặt ngưỡng thống kê hoặc GPU giờ khi chưa có cost/precision evidence. Trước run, integrator phải điền và hash tất cả:

- Exact candidate definitions, observation states, sign/tie/no-swap rules và family/block order.
- Minimum eligible queries/cell, action/objective coverage, minimum valid-rho fraction, handling constant vectors/undefined values và target precision.
- Numerical meaningful-effect/ranking/sign-precision/top-utility criteria; uncertainty estimator/resampling unit; cross-cell stability và outlier sensitivity limits; multiplicity/selection rule giữa objectives.
- Reliability gate search/evaluation split, fixed thresholds nếu có, minimum coverage và harm constraints; nếu chưa mở L2 thì đánh dấu deferred.
- Data/model/config/source/registry versions, runtime/device, expected trial count và budget caps cho wall time, GPU hours, forwards, memory/storage, retries.
- Frozen launch command, output schema, campaign run IDs, validation checklist và review date. Runtime command mới chỉ đưa vào launch plan sau khi runner có CLI thực.

Giá trị unresolved chặn scored campaign, không chặn literature/instrument work. Cost-only smoke dùng registry riêng, không dùng để chọn objective theo outcome; results exploratory và exclusions được lưu.

**Exit:** `CASE-QCGS-01`/experiment card pre-run, protocol đầy đủ bảy deliverables §39, hashes và launch eligibility checklist. Không gọi file plan này là frozen experimental protocol.

## 8. P3 — Instrument implementation, review và reproducibility gate

Module: [Instrument](../../.agents/skills/research-orchestrator/references/build-research-instrument.md). Coding worker triển khai contract P2 bằng cách tái sử dụng instrument hiện có sau audit; không viết deployable selector ở L0.

- [ ] Inspect `src/evaluation/oracle_support_utility.py`, `src/runtime/query_gradient_registry.py`, existing diagnostic/notebook paths và relevant configs; xác định phần reuse và sửa tối thiểu cần để đo exact objectives.
- [ ] Ghi action records đủ tái tính: query/action opaque keys, candidate snapshot/hash, baseline/action outcomes cho objective, evaluator CE/predictions riêng, valid/undefined reasons và costs.
- [ ] Validate formula/sign trên known cases: no-swap=0, reward/loss orientation, rank ties, constant objective, empty positive set, action ordering, random expectation và non-finite/undefined treatment.
- [ ] Regression cho label isolation, trial order independence/reset, unchanged cache, baseline exact parity và exhaustive action coverage. Dùng relevant `unittest`; CPU checks không chứng minh CUDA parity.
- [ ] Reviewer fresh context kiểm diff theo protocol và acceptance criteria; fix concrete findings rồi rerun checks bị ảnh hưởng. Không mở review loop mới sau clean review nếu không có thay đổi liên quan.
- [ ] Freeze source/config/registry/protocol hashes và commit implementation trước CUDA; record dependency/runtime deviations và reproduction command.
- [ ] Chạy real CUDA smoke trên separate smoke queries, kiểm baseline/raw/log/provenance với numerical tolerance đã định trước; không dùng utility smoke để tune objective.

Các command đã có trong repo, thực thi sau khi xác nhận data root/runtime phù hợp:

```sh
uv sync --locked --offline
source .venv/bin/activate
PYTHONPATH=src python -m runtime.preflight --data-root ~/data --dataset CIFAR100C
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

`--offline` chỉ dùng khi dependencies đã cached; thiếu cache phải được ghi nhận và xử lý trong runtime setup. Không để tests tự tải data/checkpoint hoặc launch campaign. Campaign CLI/arguments sẽ ghi đúng theo implementation, không dùng placeholder command như thể đã chạy.

**Exit:** review findings resolved, relevant CPU/regression checks và CUDA smoke có receipts; baseline parity/reset/access/exhaustiveness được chứng minh trong phạm vi smoke. Campaign vẫn cần audit raw riêng.

## 9. P4 — L0 exhaustive execution và L1 information screen

Module: [Research loop](../../.agents/skills/research-orchestrator/references/run-research-loop.md). Budget đúng launch plan P2/P3; lưu actual costs và failed attempts.

- [ ] Launch complete preregistered batches cho hai OOD cells bằng unique run IDs. Theo dõi process/job handles, attempted/completed counts, checkpoints và bounded waits; không restart từ timeout quan sát nếu handle vẫn sống.
- [ ] Đo exact objective outcomes của mọi legal action từ cùng snapshot. Lưu all actions, no-swap, controls, raw values và validity flags; bảo toàn negative/harmful examples.
- [ ] Verify provenance, registry membership/overlap, query quotas, action enumeration độc lập, finite utility và objective validity; recompute metrics từ raw thay vì tin summary runner.
- [ ] Ranking: Spearman per query (tied ranks), mean/median, positive-rho fraction, valid/total query count; Kendall chỉ thêm khi trả lời câu hỏi mới.
- [ ] Sign prediction: `P(U>0 | delta_z>0)`, số positive actions/queries, helpful/harmful counts; báo cả query-balanced và pooled nếu dùng, không coi actions cùng query là independent samples.
- [ ] Top action: argmax objective theo canonical ties/no-swap rule đã khóa; utility so no-swap, random, supervised first-order và oracle. Regret `U_oracle-U_selected`.
- [ ] Stability: hai OOD cells, corruption groups, baseline-correct/wrong, uncertainty strata và memory states. Correctness/OOD strata chỉ dùng cho evaluator diagnosis, không làm reliability gate inputs.
- [ ] Sensitivity: mean/median/trimmed mean, harmful/helpful magnitude, largest-contributor removal rule đã freeze, harm rate và uncertainty theo query hoặc stream blocks phù hợp dependency.
- [ ] Báo prediction transitions, entropy/confidence, calibration (binning/estimator đã khóa khi dùng), ID/OOD effects và cost. Không báo true-label CE cho OOD dưới nhãn known-class giả.
- [ ] Review mỗi objective/family theo gate dưới; giữ negative và undefined coverage trong reports. Không lấy pooled positive result để che cell đảo chiều.

### Decision order dùng cho mọi block

| Decision | Evidence bắt buộc / hành động |
|---|---|
| INVALID | Provenance/leakage/parity/candidate mismatch, non-finite values trái contract hoặc campaign declared complete nhưng bỏ legal actions: repair instrument, giữ invalid run, không scientific STOP |
| INCONCLUSIVE | Chưa đủ planned queries/actions, undefined-objective coverage vượt ngưỡng, intervals không đủ precision hoặc budget hết: ghi thiếu gì; extension chỉ theo plan/revision riêng |
| STOP OBJECTIVE FAMILY | Exact objectives đã kiểm trong family không có positive ranking/selected utility ổn định so controls hoặc cross-cell reversal theo numerical gates: đóng các formulations/access đã kiểm, không claim mọi entropy/label-free objective bất khả thi |
| GO ESTIMATOR | Exact channel đạt tất cả frozen ranking, selected utility/sign benefit, both-cell stability, coverage và outlier constraints: mở L4 cho channel đó; chưa gọi là deployable success |
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

Mở `CASE-QCGS-03` chỉ cho exact channel đã qua information gate. Nếu không có channel qua, phase này ghi `not reached` với upstream decision; không gọi là implementation thiếu.

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

1. P0 inventory và data/runtime probe; lưu state với trạng thái evidence thực tế.
2. Launch song song P1 literature verification và P2a read-only instrument audit bằng bounded packets, file ownership riêng.
3. Integrator reconcile, viết/freeze P2 protocol gồm đủ bảy mục §39; ghi những numerical/budget fields còn thiếu.
4. Sau freeze mới launch coding worker, rồi reviewer trên diff và campaign theo các gate P3–P4.

Việc viết plan này không tự launch workers, tests hoặc CUDA campaign. Khi thực thi, launcher tiếp tục theo authorization hiện có, không chèn approval gates vào từng variant; chỉ dừng để lấy thông tin thực sự thiếu khiến bước phụ thuộc không thể tiến hành đúng.
