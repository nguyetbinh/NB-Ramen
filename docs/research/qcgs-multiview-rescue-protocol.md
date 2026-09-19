# QCGS: một diagnostic cứu hướng bằng soft target nhiều view

**Ngày:** 2026-09-19\
**Nhánh:** `qcgs-label-free-diagnostic`\
**Base code đã đọc:** `cb3c92cded0e6ad4601b5e1f876835a9cf3992c4`\
**Trạng thái:** đã implement và kiểm thử CPU; chưa chạy CUDA smoke/Stage A/Stage B, chưa có kết quả khoa học cho biến thể này.\
**Phạm vi:** một selector chính, same-pseudo-class one-swap; Stage A 16 và Stage B 128 query mới mỗi mức OOD 0 / 0.5.

## 1. Quyết định nghiên cứu

Giữ **STOP** cho entropy-sign một view trong [protocol cũ](query-conditioned-gradient-selection.md) và [report đã kiểm chứng](../../plans/20260919-qcgs-diagnostic/reports/results.md). Không đổi kết luận cũ thành REVISE hoặc GO.

Cho phép kiểm tra một giả thuyết mới: **gradient hướng tới soft target tổng hợp từ nhiều view, tính tại trạng thái Ramen, có dự đoán được utility của một phép thay support không?**

Đây là một lần thử có giới hạn. Nếu tín hiệu mới không vượt Ramen/random hoặc không có giá trị vượt ensemble các view, dừng đầu tư vào nhánh QCGS này trong thiết lập đang xét. Không tự chuyển sang full matrix, full reranking, adaptive support size hay tìm nhiều bộ threshold.

## 2. Vì sao còn đáng thử, và bằng chứng nào chưa đủ

Evidence cũ: ZIP `qcgs-label-free-evidence.zip`, SHA-256 `b7659601e54c3862bd31b4f6917279c2326054ef98884fbf3bfdc6aef51c317c`; source thực nghiệm `8298ba434fa3e63710b899500a822a3575ca4884`.

| Quan sát đã đo | OOD=0 | OOD=0.5 | Diễn giải |
|---|---:|---:|---|
| Stage A: exact oracle CE gain trung bình | +0.408288 | +0.120676 | Có support thay thế hữu ích trong cùng không gian hành động |
| Stage B: supervised-reference CE gain | +0.174322 | +0.413015 | Khi được dùng nhãn thật, tín hiệu gradient chọn được phương án có lợi trung bình |
| Stage B: entropy-sign CE gain | −0.018072 | −0.014818 | Tín hiệu không nhãn hiện tại thất bại |
| Stage B: entropy-sign gain trên query Ramen đang sai | −0.042454 | −0.041492 | Phù hợp với cơ chế củng cố dự đoán sai; chưa phải chứng minh nhân quả riêng biệt |
| Stage B: entropy query trước → sau Ramen | 2.465684 → 1.506593 | 2.599607 → 1.516553 | Có lý do kiểm tra lại điểm tính gradient |

Phân tích hậu nghiệm Stage A cho tương quan thứ hạng giữa entropy-sign score và entropy gain thực tế chỉ khoảng 0.137 / 0.146. Ngay cả chọn swap có entropy thực tế thấp nhất trong toàn bộ không gian cũng cho CE gain −0.052351 / +0.012330. Vì vậy, sửa phép xấp xỉ entropy đơn thuần chưa giải quyết mục tiêu không khớp CE.

Lọc theo norm cũng chưa có evidence vững: nhóm norm giữa theo các bin đã khóa từ Stage A có Stage B gain +0.007043 / +0.010862, nhưng bỏ hai gain lớn nhất thì còn −0.005775 / −0.001761. Các bin khác nhau giữa hai cell; không được dùng chúng thành một gate phụ thuộc OOD thật. Chuẩn hóa toàn bộ query gradient bằng một số dương không đổi argmax hoặc gate score>0 của cùng query.

Nguồn local: [raw records](../../evidence/qcgs-kaggle-b7659601e54c/qcgs-label-free-evidence/queries.jsonl), [analysis đã audit](../../evidence/qcgs-kaggle-b7659601e54c/qcgs-label-free-evidence/analysis.json), [phép tính hậu nghiệm](../../evidence/qcgs-kaggle-b7659601e54c/post-hoc-failure-analysis.json). Evidence lớn nằm ngoài Git; các link local cần artifact tương ứng. Những phân tích này dùng dữ liệu đã xem, chỉ tạo giả thuyết, không xác nhận biến thể mới.

## 3. Một selector chính: multi-view target tại trạng thái Ramen

Gọi θ₀ là tham số gốc; S là support Ramen; G(S) là aggregate theo đúng Ramen. Giữ nguyên cập nhật:

\[
\theta_R=\theta_0-\eta\,\operatorname{sign}(G(S)),\quad \eta=0.01.
\]

### 3.1 Tạo soft target mà không dùng nhãn

Từ đúng ảnh corrupted hiện tại x, tạo V=8 view cố định theo mục 4. Dùng mô hình tại θ₀, cùng text features/class order/logit scale của Ramen:

\[
t_q=\operatorname{stopgrad}\!\left[\frac{1}{8}\sum_{v=1}^{8}
\operatorname{softmax}(f_{\theta_0}(a_v(x_q)))\right].
\]

Trung bình **probability**, không trung bình logits. Softmax và trung bình tính float32. Với CE của các ensemble control, tính log-probability bằng `logsumexp(log_softmax(view_logits), view_axis) - log(8)` để tránh log(0); không thêm epsilon/clip tùy ý làm đổi metric. Không sharpening, temperature bổ sung, confidence filtering, hard pseudo-label hoặc cập nhật teacher theo stream. Teacher là cùng CLIP đóng băng tại θ₀, không phải model mới được huấn luyện.

Các view có thể sửa dự đoán khi ảnh gốc gây nhầm, nhưng cũng có thể cùng sai. Không gọi chúng là quan sát độc lập hoặc coi agreement là chứng nhận đúng.

### 3.2 Tính query gradient tại đúng baseline cần so sánh

\[
L_t(\theta,q)=-\sum_c t_q(c)\log p_\theta(c\mid x_q),\qquad
h_t=\nabla_\theta L_t(\theta,q)\big|_{\theta_R}.
\]

Đây là gradient theo các normalization-affine parameters mà Ramen đang thích nghi. Target t_q đã detach. Không dùng entropy của t_q làm loss backward: nó đã cố định và không tạo query gradient.

Nếu chỉ lấy chính prediction hiện tại làm soft target rồi đánh giá lại cùng prediction ở cùng tham số, gradient CE có thể bằng 0. Thiết kế này lấy target từ các view tại θ₀ và student trên ảnh gốc tại θ_R; vẫn phải xử lý trường hợp gradient thực tế bằng 0 bằng no-swap.

### 3.3 Chấm điểm và chọn một swap

Giữ candidate space cũ: top-k=5 mỗi pseudo-class; pool tối đa M=10 giữ chính xác các support baseline; bỏ một support và thêm một extra cùng pseudo-class. Chỉ một swap trên toàn bộ support set của query.

Với mỗi S′, giữ nguyên entropy/distance weights và active-class divisor của Ramen, tính aggregate G(S′), rồi:

\[
\delta\theta=\theta_{S'}-\theta_R
=-\eta[\operatorname{sign}(G(S'))-\operatorname{sign}(G(S))],
\]

\[
\boxed{s_t(S\to S')=-h_t^\top\delta\theta
=\eta h_t^\top[\operatorname{sign}(G(S'))-\operatorname{sign}(G(S))].}
\]

Chọn argmax nếu score **>0**; nếu không thì giữ S. No-swap thắng tại 0; positive ties theo canonical action order. Không thêm gate norm, confidence hoặc agreement vào selector chính.

Score xấp xỉ giảm soft-target CE, chưa bảo đảm giảm true-label CE. Thay điểm tính gradient từ θ₀ sang θ_R là giả thuyết cải thiện xấp xỉ, không phải một bug đã được chứng minh của protocol cũ.

**Khi xác minh S′, luôn reset về θ₀ rồi áp dụng đúng một bước −η·sign(G(S′)). Không cập nhật thêm một bước từ θ_R.** Primary output là prediction trên ảnh gốc x_q sau swap, không phải output ensemble. Sau mỗi trial reset tham số/optimizer; continuing output/cache trajectory vẫn là Ramen.

## 4. Khóa view và các bất biến của baseline

Để có một thử nghiệm xác định, chọn trước `four-corners-30-flip-v1` cho ảnh RGB CIFAR 32×32 đã corrupted:

1. Crop 30×30 ở bốn góc theo thứ tự `(left, top)`: `(0,0)`, `(2,0)`, `(0,2)`, `(2,2)`; hộp crop `[left, top, left+30, top+30)`.
2. Với mỗi crop: view không flip rồi view horizontal flip, tổng cộng 8 view.
3. Áp dụng preprocessing CLIP hiện có cho từng view; ảnh gốc qua preprocessing cũ không đổi.

Không lấy ảnh clean, corruption khác của cùng base image, nhãn hoặc metadata oracle để tạo view. Crop trước resize/normalize, không crop tensor đã normalize. Đây là lựa chọn nhẹ và có thể tái tạo, chưa được chứng minh tối ưu hoặc luôn giữ nhãn; khóa Pillow/CLIP/torchvision, interpolation/antialias và hash input/view tensors trong preflight. Không đổi view sau khi nhìn utility.

| Thành phần | Giá trị giữ nguyên |
|---|---|
| Dataset/model | CIFAR-100-C severity 5 / CLIP ViT-B/16 |
| Split | 80 ID / 20 OOD, `open-set-cifar100-split-v1` |
| Cells/stream | OOD 0 và 0.5; block=64; seed=0 |
| Source budget/batch | 400 samples/domain; B=100 |
| Memory/retrieval | 750/pseudo-class; k=5; M=10; beta=5 |
| Update | SignSGD, lr=0.01, momentum=0, weight decay=0 |
| Arithmetic | Baseline aggregate dtype/order giữ nguyên; lấy sign trước cast float32 để chấm score |

Giữ batch-atomic visibility: thêm batch gốc trước retrieval, self-support được giữ. **Không thêm view vào memory**, không refresh cached gradients bằng h_t và không chọn lại support cho từng view. Cả hai OOD cell dùng cùng thuật toán/view/gate; selector không nhận OOD ratio hoặc ID/OOD flag.

By-sample normalization gắn tham số với từng vị trí batch. Teacher, gradient tại θ_R và trial phải giữ mapping vị trí query; không flatten 8×B thành batch mới rồi mặc định tham số tự khớp. Thực hiện tuần tự từng view trên batch gốc; kiểm tra baseline replay sau các forward/backward phụ.

## 5. Đối chứng để phân biệt nguồn cải thiện

| Policy | Query signal / prediction | Vai trò |
|---|---|---|
| Ramen | Giữ S; prediction ảnh gốc tại θ_R | Baseline chính |
| Entropy-θ₀ | Score entropy-sign cũ, gradient tại θ₀ | Mốc so sánh tín hiệu cũ |
| Entropy-θ_R | Cùng entropy loss, gradient tại θ_R | Tách tác dụng đổi điểm tính gradient |
| **MV-target-θ_R** | h_t và score ở mục 3 | **Selector chính duy nhất** |
| Random-forced | Uniform trên mọi legal swap | Giữ đối chứng random cũ |
| Random-matched | Dùng cùng random draw khi primary swap; no-swap khi primary no-swap | So sánh chất lượng chọn, kiểm soát khác biệt abstention |
| Frozen-MV | t_q là phân phối dự đoán cuối, không adaptation | Đo giá trị teacher/augmentation riêng |
| Ramen-MV | Mean probability trên cùng 8 view tại θ_R, giữ nguyên S từ ảnh gốc | Đối chứng ensemble mạnh |
| Supervised-θ_R | Gradient true-label CE tại θ_R; evaluator-only | Reference cho chất lượng hướng tại cùng anchor |

Supervised reference chọn best positive score với no-swap tại ≤0, cùng tie rule. Nó không phải exact oracle. Không trộn số của reference mới với reference cũ tại θ₀ (vốn forced best legal swap).

Random dùng một draw/query từ RNG riêng seed=0; cả hai random policy chia sẻ draw đó, và vẫn tiêu thụ draw khi primary abstain để tránh thay RNG trajectory. Khóa mọi label-free action trước supervised backward/utility.

Ramen-MV dùng chính 8 view đã khai báo, không retrieval/adaptation riêng mỗi view. Nếu primary chỉ thắng Ramen ảnh gốc nhưng không thắng Ramen-MV, chưa đủ lý do đầu tư vào support selection. Không chuyển Frozen-MV hoặc Ramen-MV thành primary sau khi thấy kết quả.

Ngân sách primary ngoài baseline gồm 8 teacher-view forwards, forward/backward soft-target tại θ_R và verification forward cho lựa chọn nếu cần. Ramen-MV có 8 forward tại θ_R. Các control/evaluator có chi phí riêng; báo wall time, forwards, backwards và peak memory, **không mặc định bằng latency chỉ vì đều có 8 view**. Chỉ tính các side branches trên query/batch được registry chấm, không thay stream history.

## 6. Thực thi theo hai bước và dừng sớm

1. **CPU tests/preflight → CUDA smoke 2 query/cell.** Smoke kiểm tra mechanics, không chọn view/score dựa theo gain. Khóa implementation source, model/data hashes, view transform, RNG, parameter order, config và protocol trước Stage A.
2. **Stage A: 16 ID query mới/cell.** Tính score mọi legal action; xác minh true CE, soft-target CE và entropy trên ảnh gốc cho toàn bộ swap + no-swap. Tính Frozen-MV/Ramen-MV một lần/query, không chạy 8-view ensemble cho mọi swap. Exact oracle chỉ tối ưu true-label CE trong không gian one-swap trên ảnh gốc.
3. **Chỉ nếu Stage A qua gate bên dưới: Stage B 128 ID query mới/cell.** Xác minh hợp các action đã chọn bởi các policy, deduplicate sau selection. Tính các ensemble control. Stage B không exhaustive: ghi best-verified subset, exact oracle/regret phải null.

Frozen-MV và Ramen-MV là prediction controls, không phải legal swap. Không đưa chúng vào exact oracle, best-verified subset hoặc regret của không gian one-swap.

Không chạy matrix. Bắt đầu scan outcome-blind với 600 stream rows; tăng gấp đôi, cap 6.000, chỉ để đủ quota mới. Khóa prefix ngắn nhất theo nguyên batch phủ quota A/B trước scored execution. Nếu hết cap vẫn thiếu query thì INCONCLUSIVE/blocked preflight, không tái sử dụng query cũ.

Timeout mặc định 3.600 giây/cell, kế thừa runner. Có thể điều chỉnh dựa trên chi phí smoke trước Stage A, ghi vào launch lock; không điều chỉnh theo gain hoặc dùng timeout để bỏ query xấu. Chưa có ước tính giờ GPU đáng tin cho biến thể này.

### Query registry và chống dùng lại dữ liệu đã xem

- Exclusion = historical oracle exclusions + toàn bộ smoke/Stage A/Stage B QCGS cũ + smoke của lần mới + mọi query khác đã dùng để chọn thiết kế này. Lưu nguồn và SHA-256 từng tập; không chỉ đổi seed.
- Base identity là dataset fingerprint + original CIFAR `sample_idx`, gộp qua mọi corruption/severity/cell. Các view của cùng ảnh không phải query mới.
- Chia trước bằng SHA-256 UTF-8 `qcgs-multiview-rescue-v1|<dataset_fingerprint>|<sample_idx>`, integer modulo 9: bucket 0 → A, còn lại → B. Không tìm salt khác.
- Sau exclusions, lấy các ID query eligible đầu tiên trong stage đã gán, duy nhất trong mỗi cell. A/B không trùng base image qua cả hai cell. Trùng giữa hai OOD cell trong cùng stage được phép, phải báo số lượng và không coi là replicate độc lập.
- Registry chỉ chọn query được chấm; các ảnh excluded/non-scored vẫn tham gia cache lịch sử theo baseline. Không dùng correctness, norm, entropy, score hay gain để chọn query.
- Commit registry/config/launch locks trong evidence ledger trước Stage A; giữ Stage B utilities chưa mở đến khi quyết định GO_CONFIRM. Không cập nhật công thức/threshold sau A. Nếu đổi khoa học, đó là protocol mới với confirmation mới.

## 7. Metrics và decision gates mới

Với primary MV-target-θ_R trên cùng query, định nghĩa:

\[
U=CE_R-CE_Q,\quad D=CE_{RandomMatched}-CE_Q,\quad E=CE_{RamenMV}-CE_Q.
\]

U/D/E dương đều có lợi. **E là gate bổ sung cho lần thử mới**, để đòi hỏi lợi ích vượt ensemble; nó không sửa gate hoặc kết luận của thí nghiệm cũ. Tính các mean trên toàn bộ query đã đăng ký, gồm no-swap; không chỉ báo subset đã được thay support.

### Gate Stage A — quyết định có chi ngân sách confirmation không

Áp dụng theo thứ tự:

1. **INVALID:** vi phạm isolation/parity/reset/provenance/numerics; sửa mechanics trước khi suy luận khoa học.
2. **INCONCLUSIVE:** thiếu quota/outcome hợp lệ vì nguyên nhân vận hành; chỉ resume cấu hình đã khóa.
3. **GO_CONFIRM:** mean exact-oracle U>0, mean primary U>0, mean D>0 và mean E>0 ở **từng cell**; ít nhất 3 primary replacements/cell; mean per-query Spearman(score, exact true utility)>0 ở từng cell, có ít nhất 8/16 query có rho xác định.
4. **STOP:** mọi kết quả hoàn chỉnh, hợp lệ còn lại. Không tự mở rộng quota hoặc chuyển control thành primary.

Đây là gate đầu tư bảo thủ với 16 query/cell, có thể bỏ lỡ hiệu ứng thật. Nó không phải kiểm định thống kê chứng minh hiệu ứng bằng 0. Spearman tính riêng từng query trên toàn bộ legal swaps, bỏ no-swap, dùng average ranks khi tie. Báo đầy đủ cả các trường hợp rho undefined; không thay null bằng zero và không gộp các swap thành quan sát độc lập.

### Gate Stage B — quyết định có tiếp tục QCGS không

1. **INVALID** nếu integrity fail.
2. **INCONCLUSIVE** nếu incomplete; không dùng thiếu dữ liệu làm bằng chứng fail khoa học.
3. **GO_PILOT** chỉ khi mean U, D, E đều >0 trong từng cell; mỗi metric vẫn có mean>0 sau bỏ hai quan sát lớn nhất của chính nó; ít nhất 3 primary replacements/cell; tổng `wrong_to_correct - correct_to_wrong` so với Ramen không âm khi cộng hai cell. Resolve ties khi bỏ hai quan sát bằng stable query identity đã khóa.
4. **STOP** cho protocol rescue này trong mọi kết quả hoàn chỉnh, hợp lệ còn lại. Không có vòng REVISE tự động hoặc đổi gate sau kết quả.

GO_PILOT chỉ cho phép cân nhắc pilot phương pháp mới; chưa là chứng minh tổng quát, chưa tự chạy matrix. STOP không chứng minh mọi QCGS label-free đều bất khả thi.

Báo mean/median/10% symmetric trim, harmful/helpful rates và độ lớn hai nhóm, CE/accuracy transitions, coverage/no-swap reasons, soft-target CE gain, entropy gain. Báo chênh lệch primary với cả hai entropy controls, random-forced và Frozen-MV. Đặc biệt kiểm tra query Ramen đang sai: teacher có tăng xác suất nhãn thật không, có sửa prediction không; các phép này chỉ thuộc evaluator.

Kế thừa bootstrap block từ protocol cũ: block=`floor(timestep/64)`, seed=917, 10.000 draws; CI 95% khi ≥4 scored blocks/cell, query-weighted means, giữ pairing U/D/E, báo leave-one-block-out. Không coi block độc lập tuyệt đối vì chung cache history; một seed/stream không xác nhận generalization. Không suy CI cho các view như thể chúng là query độc lập.

## 8. Kiểm thử và evidence bắt buộc

- **Baseline parity:** không chạy các side branch thì output/cache/hash như Ramen cũ; sau teacher/backward phụ vẫn replay đúng baseline. S′=S cho θ và logits baseline.
- **Anchor/update:** h_t và entropy-θ_R đo tại θ_R; mọi candidate bắt đầu θ₀ và chỉ một bước; reset cả khi exception. Không dùng helper đang bọc `no_grad` để lấy gradient mới.
- **Label isolation:** đổi true label/is_ood/domain metadata không đổi views, teacher, scores hoặc label-free actions. Oracle chỉ chạy sau selection. OOD query không được gán known-class CE giả.
- **View/slot consistency:** crop/order/flip/preprocess/hash cố định; không cross-query mixing, không thêm view vào cache; không để 8-view pass làm hỏng by-sample parameter row mapping.
- **Loss/score arithmetic:** t_q detach, probability sum≈1; CE đúng `-sum(t*log_softmax(logits))`; finite-difference trên mô hình nhỏ thật có trọng số cố định kiểm tra dấu/anchor của score; không đòi first-order score bằng exact utility trên CLIP phi tuyến.
- **Numerical guards:** zero query norm → valid no-swap; missing/shape mismatch/nonfinite gradient, logits, target, aggregate, score hoặc utility → INVALID. Sign lấy ở aggregate dtype gốc; dot product float32; parameter order/hash được khóa.
- **Selection/RNG:** no-swap tại ≤0, canonical tie, random draw không đổi do verification dedup/abstention; test modified query metadata và permutations có canonicalization.
- **Registry/restore:** không trùng base image A/B hoặc exclusions; replay đúng earliest eligible assignment; resume chỉ nhận cùng source/config/view/registry hashes, không merge các protocol.

Giữ output gọn: `queries.jsonl`, `analysis.json`, locks/provenance, run logs và ZIP checkpoint atomic. Mỗi query giữ teacher probability vector và SHA, ensemble log-probability vectors, view hashes, anchor/parameter hashes, baseline và control predictions/CE, all candidate scores + legal keys, selected action/reason, RNG draw, verified utilities, coverage và cost. Stage A lưu tất cả verified swaps; Stage B chỉ selected union. Không cần dump toàn bộ gradient tensors cho mỗi candidate.

Analysis phải tái tính U/D/E, transitions, oracle/subset và gates từ raw records; scientific report ghi rõ điều gì đã chạy. Notebook được tạo không phải evidence CUDA.

## 9. Điểm triển khai trong repo và handoff

Tái sử dụng các ranh giới module hiện có:

- [query_gradient_selection.py](../../src/methods/query_gradient_selection.py): pure scorer nhận h_t và candidate aggregates, vẫn không nhận labels/provenance. Hỗ trợ anchor mới qua input, không tự lấy true labels.
- [QueryGradientUtilityProbe.py](../../src/methods/QueryGradientUtilityProbe.py): thêm teacher views, differentiable θ_R forward và các control; baseline/candidate reset không đổi.
- [query_gradient_registry.py](../../src/runtime/query_gradient_registry.py): namespace/exclusions mới, không phá registry replay của protocol cũ.
- [query_gradient_utility.py](../../src/evaluation/query_gradient_utility.py): thêm soft-target loss, ensemble controls, metrics E, gates và schema riêng.
- [query_gradient_campaign.py](../../src/runtime/query_gradient_campaign.py): Stage A → gate → Stage B, fresh evidence root/locks, checkpoint và offline audit.
- [CIFAR100C loader](../../src/datasets/corruption/CIFAR100C.py): cung cấp raw corrupted image cho side views qua ranh giới dataset phù hợp; output baseline preprocessing không đổi.
- [Kaggle builder](../../notebooks/kaggle/build-qcgs-notebook.py): sau khi code/tests hoàn tất, tạo notebook rescue riêng với source pin mới, dùng lại pipeline Hugging Face. Giữ nguyên notebook single-view đã tạo evidence STOP.

Mode riêng: `--diagnostic qcgs-multiview`, schema 2, registry namespace `qcgs-multiview-rescue-v1`. Lệnh `--diagnostic qcgs` giữ protocol cũ.

- [Notebook rescue](../../notebooks/kaggle/kaggle-qcgs-multiview.ipynb): source nhúng và cố định; bật Internet/GPU, chạy từ trên xuống hoặc Save & Run All. Xuất `qcgs-multiview-evidence.zip`; Stage B chỉ chạy sau GO_CONFIRM đã commit trong ledger.
- [Trạng thái triển khai và validation](../../plans/20260919-qcgs-multiview-rescue/reports/results.md).
- Selector/view: [pure helpers](../../src/methods/query_gradient_multiview.py), [probe](../../src/methods/MultiViewQueryGradientUtilityProbe.py); [audit/gates](../../src/evaluation/query_gradient_multiview.py); [config và exclusions](../../cfg/research/query-gradient-multiview/).

```sh
python scripts/run-oracle-support-utility.py --diagnostic qcgs-multiview \
  --preflight-only --evidence-dir /tmp/qcgs-multiview-cpu
# Trên CUDA, source checkout sạch; thay đường dẫn dữ liệu/evidence tương ứng:
python scripts/run-oracle-support-utility.py --diagnostic qcgs-multiview \
  --execute --data-root /tmp/nb-ramen-qcgs-mv-data \
  --evidence-dir /kaggle/working/qcgs-multiview-evidence
# Thêm --resume để tiếp tục đúng campaign/source/environment đã khóa.
python scripts/run-oracle-support-utility.py --diagnostic qcgs-multiview \
  --audit --evidence-dir /kaggle/working/qcgs-multiview-evidence
```

CPU tests trên mô hình nhỏ thật xác minh mechanics, không thay thế CUDA smoke trên CLIP. Registry khoa học được tạo từ scan đúng dataset/model trên CUDA; unit-test registries không phải query registry để thu evidence.

Task triển khai có thể dùng trực tiếp:

> Thực hiện `docs/research/qcgs-multiview-rescue-protocol.md` trên nhánh `qcgs-label-free-diagnostic`, không dùng skill ak. Giữ mọi thay đổi không liên quan. Implement một primary MV-target-θ_R và các control đã khóa; không sửa STOP/protocol/evidence cũ. Chạy CPU tests/preflight, tạo notebook Kaggle source-pinned nếu local không có CUDA; notebook dừng sau Stage A nếu không GO_CONFIRM. Dùng query registry mới loại mọi ảnh đã xem, checkpoint và ZIP evidence. Chỉ báo kết quả thực sự đã chạy, audit raw metrics theo gates mới; dừng sau diagnostic, không mở full matrix.

## 10. Liên hệ nghiên cứu và giới hạn

[MEMO](https://arxiv.org/abs/2110.09506) dùng entropy của phân phối trung bình qua augmentations; [TPT](https://arxiv.org/abs/2209.07511) dùng nhiều view trong test-time prompt tuning của VLM. Chúng tạo động cơ kiểm tra thông tin từ view, không xác nhận công thức support-swap ở đây. [MTA](https://arxiv.org/abs/2405.02266) cho thấy test-time augmentation không cần prompt learning vẫn là đối chứng mạnh. Vì vậy, thêm nhiều view tự nó không phải đóng góp mới của QCGS.

[Realistic Evaluation of TTA](https://arxiv.org/abs/2407.14231) phân tích khó khăn của lựa chọn không nhãn; confidence/consistency không phải bảo đảm về accuracy. Soft teacher ở đây có thể sai có hệ thống, và không có nhãn để xác nhận nó online.

Đóng góp chỉ đáng theo đuổi nếu một tín hiệu không nhãn **chọn được support hữu ích hơn đối chứng**, vượt phần lợi ích của ensemble, với chi phí được báo trung thực. Report này khóa một phép thử cho giả thuyết đó, không tuyên bố đã cứu được hướng hoặc đã thiết lập tính mới.
