---
phase: 1
title: "Chốt câu hỏi và protocol"
status: pending
priority: P1
dependencies: []
---

# Phase 1: Chốt câu hỏi và protocol

## Overview

Khóa thí nghiệm nhỏ nhất có thể phân biệt chọn support hữu ích với chỉ làm mô hình tự tin hơn. Các lựa chọn dưới đây là đề xuất nghiên cứu, chưa phải cấu hình đã được người dùng chốt.

## Context

- [Pilot B](../20260907-oracle-support-utility/reports/pilot-b-completed.md) có 128 query/cell tại OOD=0 và 0.5; oracle sửa thêm hai dự đoán mỗi cell. Kết quả dùng nhãn query thật, không chứng minh selector không nhãn.
- [Oracle diagnostic](../../docs/research/oracle-support-utility-probe.md) giữ cache và SignSGD của Ramen, chỉ hoán đổi một support cùng pseudo-class.
- [Ramen](../../src/methods/Ramen.py) đã tính entropy gradient cho từng query ở pretrained parameters trước retrieval. Có thể tái sử dụng gradient này; scoring vẫn phát sinh compute.
- Các link trong phase dùng mốc thư mục plan; source/code mới chỉ là phạm vi đề xuất.

## Requirements

- Dùng ảnh, dự đoán, feature và gradient không nhãn trong selector; true label/domain/ID-OOD chỉ dùng evaluator.
- Giữ batch visibility, cache admission, self-support eligibility, weighting, class divisor và reset giống baseline.
- Pilot khởi đầu đề xuất: CIFAR100C, ViT-B/16, k=5, M=10, batch=100 và các hằng số đã có trong oracle contract; không khóa thêm ngưỡng hiệu quả tùy ý.
- Ghi rõ task hiện tại là chọn support hỗ trợ ID query dưới open-world stream; không tự mở rộng thành nhận diện lớp unknown.

## Files

- Read: `src/methods/Ramen.py`, `src/methods/OracleSupportUtilityProbe.py`, `src/evaluation/oracle_support_utility.py`, `tests/test_oracle_support_utility.py`.
- Read: `docs/research/gradient-compatibility-research-direction.md`, `docs/research/oracle-support-utility-probe.md` và các báo cáo A/B.
- Create: `docs/research/query-conditioned-gradient-selection.md` làm specification sau khi protocol được chốt; `reports/protocol-and-related-work.md` trong plan này.
- Không sửa source/config trong phase này.

## Implementation Steps

1. Ghi giả thuyết: một score không nhãn sẽ chọn được thay đổi support có supervised query utility cao hơn random trên candidate pool cố định.
2. Định nghĩa utility đo thật là CE của Ramen trừ CE sau cập nhật; accuracy và số đúng→sai/sai→đúng là kết quả riêng. OOD query chỉ được chấm bằng metric open-set phù hợp, không dùng CE known-class giả.
3. Chốt query không nhãn ban đầu là entropy gradient; phân biệt với supervised gradient trong probe. Multi-view là nhánh thử sau nếu gradient một ảnh yếu.
4. Dành các query/domain/stream seed riêng cho phát triển và đánh giá xác nhận; seed khác chưa bảo đảm ảnh khác. Ghi image IDs và kiểm tra trùng lặp, không tune trên bộ đánh giá cuối.
5. Rà paper liên quan theo ba nhóm: Ramen/sample selection; query-conditioned gradient utility/data selection; entropy/multi-view test-time adaptation. Lập bảng khác biệt về labels, cache, update rule và query conditioning trước claim novelty.
6. Khóa sampling budget, metric chính, tie handling, numerical tolerance, random seed và gate trước khi xem kết quả mới. Ước lượng runtime bằng pilot nhỏ trước khi cam kết GPU budget.

## Primary reading

- [Ramen](https://arxiv.org/abs/2604.21728): embedding-gradient cache và active sample selection theo domain consistency/prediction balance.
- [TPT](https://arxiv.org/abs/2209.07511): entropy với confidence selection trên augmented views. Đây là cơ sở tham khảo cho nhánh multi-view, không phải bằng chứng QCGS có hiệu quả hoặc mới.
- Hai nguồn đã đối chiếu ngày 2026-09-14; đây chưa phải literature review đầy đủ.

## Todo

- [ ] Viết protocol và split registry.
- [ ] Viết bảng related work, phân biệt proposal với kết quả đã có.

## Success Criteria

Có specification đủ để thực hiện phase 2 mà không cần chọn ngưỡng theo kết quả test. Mỗi claim trong protocol có nguồn hoặc được ghi rõ là giả thuyết.

## Risks and rollback

Pilot cũ chịu screening và khác lịch sử cache giữa hai OOD ratios. Dùng làm motivation; không dùng làm causal OOD evidence. Nếu protocol thiếu dữ liệu độc lập, sửa cách chia dữ liệu trước khi chạy, giữ artifacts cũ nguyên vẹn.
