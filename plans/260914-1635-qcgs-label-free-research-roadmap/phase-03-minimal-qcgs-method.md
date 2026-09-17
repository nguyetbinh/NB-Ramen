---
phase: 3
title: "Minimal QCGS method"
status: pending
priority: P2
dependencies: [2]
---

# Phase 3: Minimal QCGS method

## Overview

Sau gate diagnostic, triển khai QCGS-Ramen v0 trả dự đoán thật từ support do selector không nhãn chọn. v0 là one-swap, không phải rerank toàn bộ top-M→top-k.

## Context and architecture

[Diagnostic](./phase-02-label-free-utility-diagnostic.md) cung cấp scorer đã kiểm chứng. Luồng: query feature + entropy gradient → cache admission như Ramen → baseline top-k và extras top-M → chọn tối đa một swap → aggregate → SignSGD → inference → reset.

Giữ k=5, M=10 cho phiên bản khởi đầu trừ khi phase 1 ghi quyết định khác. Nhóm ít mẫu vẫn giữ toàn bộ support sẵn có; không drop cả class do thiếu candidate hoặc score âm.

## Requirements

- Độc lập với OracleSupportUtilityProbe, không kế thừa hook nhận nhãn thật.
- Không thay cache admission/memory content, confidence weighting, distance weighting, batch visibility hoặc self-support eligibility.
- Với selector tắt, không có swap hoặc fallback, khớp Ramen về selected indices, aggregate, logits và reset trên cùng runtime.
- Top-k baseline lấy riêng, không giả định prefix top-M giống top-k khi tie. Candidate universe phải giữ baseline membership.
- Log chi phí scoring và workspace; reuse query entropy gradient không đồng nghĩa tổng chi phí bằng Ramen.

## Files

- Create: `src/methods/QueryGradientRamen.py`, `tests/test_query_gradient_ramen.py`, `cfg/research/query-gradient-selection/`.
- Modify: `src/methods/query_gradient_selection.py` nếu cần, `src/methods/__init__.py`, `docs/research/query-conditioned-gradient-selection.md`.
- Read: `src/models/ModelForBySampleTTA.py`, `src/methods/Ramen.py`, `src/methods/ConsensusRamen.py`.
- Không thêm QCGS vào canonical matrix hiện có trong phase này.

## Implementation Steps

1. Đăng ký method riêng; truyền dữ liệu không nhãn cho scorer đã qua phase 2.
2. Dùng cached gradients có cùng parameter ordering/scale như Ramen; giữ số samples mỗi lớp và divisor đúng baseline.
3. Nếu không có swap score dương hợp lệ, giữ nguyên update Ramen; đây không phải skip adaptation.
4. Chunk candidate scoring để tránh materialize toàn bộ số swap × gradient dimension khi bộ nhớ lớn. Chỉ tối ưu sau khi đối chiếu scalar arithmetic/ranking.
5. Verify deterministic ties, sparse caches, zeros/nonfinite gradients, multiple queries, output order, reset khi exception và parity khi selector không đổi support.
6. Chạy pilot đã khóa so với NoAdapt, Ramen, Consensus và random one-swap; đo latency và peak memory trên cùng thiết bị/batch.

## Todo

- [ ] `pytest -q tests/test_query_gradient_ramen.py` rồi `pytest -q tests/test_ramen_memory_bytes.py tests/test_ramen_cpu_half.py tests/test_consensus_ramen.py tests/test_oracle_support_utility.py`; thêm `tests/test_ramen_cuda_half.py` khi có CUDA.
- [ ] Chạy integration/syntax checks cho registry/config và real-data parity smoke.

## Success Criteria

Output đúng contract, không có label access, fallback khớp baseline, support count không đổi và báo cáo runtime thật. Hiệu quả so với baseline cần kết quả thực; việc tests pass không chứng minh phương pháp tốt hơn.

## Risks and rollback

Optimizing individual candidate similarity không tương đương tối ưu aggregate SignSGD. Không thay scorer đã kiểm chứng bằng cosine vì tiện triển khai. Giữ method/config opt-in để rollback bằng cách chọn Ramen, không sửa baseline đã khóa.
