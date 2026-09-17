---
phase: 4
title: "Mechanism and robustness"
status: pending
priority: P2
dependencies: [3]
---

# Phase 4: Mechanism and robustness

## Overview

Kiểm tra lợi ích có đến từ việc dùng đúng query để chọn support, và có lặp lại ngoài stream block không. Tham chiếu [v0 contract](./phase-03-minimal-qcgs-method.md) và [Pilot B limitations](../20260907-oracle-support-utility/reports/pilot-b-completed.md).

## Requirements

- So sánh từng yếu tố trên cùng query, stream order, source snapshot và cache timeline.
- Development và confirmation độc lập theo protocol, không chọn best seed.
- Labels dùng để phân nhóm kết quả đúng/sai và ID/OOD chỉ trong evaluator.

## Baselines and ablations

| Kiểm tra | Mục đích |
|---|---|
| NoAdapt, Ramen, Consensus | Hiệu quả so với baseline đã có |
| Random one-swap, cosine one-swap | Selector có vượt chọn tùy ý và gradient matching đơn giản không |
| Query-only entropy SignSGD | Memory selection có giá trị hơn chỉ dùng gradient query không |
| Shuffled-query score, feature-only/entropy-only support ranking | Tác dụng của đúng query-gradient so với các tín hiệu sẵn có |
| Same-cache self-support exclusion ở cả baseline và QCGS | Gain có bị chi phối bởi query tự chọn chính mình không |
| Sign-aware vs cosine; one-swap vs all-class rerank | Update rule và tương tác giữa nhiều support |
| Paired batch-atomic vs causal variants | Tách query conditioning khỏi thay đổi stream visibility |

Self-exclusion và causal chỉ là ablation riêng, không đổi primary protocol. Giữ candidate/history hợp lệ theo từng semantics; shuffled query không lấy thông tin tương lai vượt protocol. Với multi-view, thêm query-only và baseline tương ứng cùng compute budget.

## Files

- Create: `scripts/run-query-gradient-pilot.py`, `cfg/research/query-gradient-ablation/`, `tests/test_query_gradient_pilot.py`, `reports/mechanism-and-runtime.md` trong plan này.
- Modify: `src/methods/QueryGradientRamen.py`, `src/methods/query_gradient_selection.py`, `tests/test_query_gradient_ramen.py` chỉ cho ablation cần thiết.
- Read: `src/runtime/experiment_matrix.py`, `src/runtime/consensus_ablation_matrix.py`, `src/methods/SupportAblations.py`.

## Implementation Steps

1. Khởi đầu block × OOD={0,0.5}; sau tín hiệu ban đầu, bổ sung iid_mixed/recurring và các seed xác nhận đã chốt. Báo cáo distribution của paired delta, không chỉ mean gộp.
2. Thực hiện baseline trọng yếu: Ramen, Consensus, random, query-only và shuffled-query. Mỗi ablation hỏi một câu; không chạy tổ hợp mọi heuristic.
3. Phân tích query baseline đúng/sai, confidence, cache occupancy, support self/history, entropy gain vs CE gain, đúng→sai/sai→đúng và score-vs-utility.
4. Dùng uncertainty theo seed/stream hoặc block resampling phù hợp phụ thuộc thời gian; không coi từng query trong cùng stream là lần chạy độc lập.
5. Thử rerank toàn bộ M→k chỉ sau one-swap đã có tín hiệu. Đánh giá lại utility của aggregate vì nhiều lựa chọn tốt riêng lẻ có thể xung đột khi gộp.
6. Nếu muốn claim OOD gây hại, thiết kế paired contamination riêng giữ ID query/domain/history cơ sở phù hợp và kiểm soát thay đổi cache do OOD. Nếu chưa làm, chỉ mô tả độ ổn định ở các OOD ratios.
7. Đo compute chính method riêng với diagnostic oracle, đồng bộ thiết bị khi timing; ghi cost theo gradient dimension/candidate count.

## Todo

- [ ] Validate run identities, paired fingerprints, deterministic controls và schema bằng focused tests.
- [ ] Viết bảng kết quả, ablation, phạm vi claim và quyết định mở rộng/dừng.

## Success Criteria

Có bằng chứng lặp lại cho phần giá trị gia tăng của support selection và query-conditioning; accuracy không bị che bởi entropy/CE giảm. Nếu chỉ hơn Ramen mà không hơn query-only hoặc shuffled-query, sửa claim và điều tra trước khi full matrix. Nếu chỉ tốt ở block, thu hẹp claim thay vì mô tả robustness phổ quát.

## Risks and rollback

Các ablation dễ tăng chi phí mà không giải quyết câu hỏi chính. Ưu tiên controls thiết yếu, chỉ tăng M hoặc số swap sau profiling. Giữ primary config riêng, mọi thử nghiệm khác có identity/version riêng.
