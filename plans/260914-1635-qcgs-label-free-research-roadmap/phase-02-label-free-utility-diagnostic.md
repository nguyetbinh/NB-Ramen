---
phase: 2
title: "Label-free utility diagnostic"
status: pending
priority: P1
dependencies: [1]
---

# Phase 2: Label-free utility diagnostic

## Overview

Thay gradient supervised trong phép xếp hạng bằng gradient entropy query, sau đó để evaluator đo utility thật. Mục tiêu là kiểm chứng tín hiệu trước khi xây phương pháp hoàn chỉnh.

## Context and architecture

Tham chiếu [oracle contract](../../docs/research/oracle-support-utility-probe.md) và [phase 1](./phase-01-start.md).

Với support set S của Ramen, G(S) là gradient gộp đúng weighting/divisor. Với một swap hợp lệ tạo S', đặt h_q = gradient entropy query tại pretrained parameters. Score đề xuất:

`score(q, S→S') = lr * dot(h_q, sign(G(S')) - sign(G(S)))`.

Score xấp xỉ lợi ích trên entropy so với bước Ramen, không bảo đảm giảm supervised loss. Dùng hướng SignSGD của **tổng gradient** vì cosine từng support không mô tả đầy đủ tác động của swap. Xấp xỉ bậc nhất phải được đối chiếu bằng forward thật; không coi positive score là chứng minh đúng nhãn.

## Requirements

- Chọn action bằng dữ liệu không nhãn trước; evaluator mới chấm bằng nhãn. Oracle xếp hạng riêng, không định hình candidate pool của selector.
- Các phương pháp nhìn cùng query, cache snapshot, baseline top-k và extras trong top-M; giữ min(k, class-size) support mỗi lớp.
- Chấm no-swap, entropy-sign score, entropy-gradient cosine preference và random hợp lệ. Oracle supervised giữ làm tham chiếu.
- Nếu dữ liệu cũ không lưu h_q hoặc đầy đủ gradient candidates, rerun query đã định trước; không dựng score không nhãn từ supervised scalar đã lưu.

## Files

- Modify: `src/methods/OracleSupportUtilityProbe.py`, `src/evaluation/oracle_support_utility.py`, `scripts/run-oracle-support-utility.py`, `tests/test_oracle_support_utility.py`.
- Create: `src/methods/query_gradient_selection.py` cho phép chọn thuần không nhãn có thể tái sử dụng ở phase 3; `cfg/research/query-gradient-diagnostic/`; `reports/label-free-pilot.md` trong plan này.
- Source filenames theo snake_case/PascalCase sẵn có trong module Python; script/config/docs theo quy ước repo.
- Giữ mode/schema oracle cũ tương thích; thêm version mới nếu sidecar contract thay đổi.

## Implementation Steps

1. Tách function chọn swap chỉ nhận gradients/weights/distances và dữ liệu không nhãn; không nhận labels hoặc provenance evaluator.
2. Sao chép h_q trước khi gradient bị thay bởi aggregate/update. Tính cosine và sign-aware score trên cùng swap space.
3. Chọn tối đa một swap có score tốt nhất vượt no-swap=0; hòa điểm ưu tiên no-swap rồi thứ tự cố định. Không có swap, gradient không hợp lệ hoặc không có cải thiện dự kiến thì giữ Ramen.
4. Chạy một tập nhỏ exhaustive đề xuất 16 query mới để kiểm chứng arithmetic và ranking. Với pilot lớn hơn đề xuất 128 ID query/cell ở OOD=0 và 0.5, verify các action selector chọn cộng một mẫu swap random được chọn độc lập với score/oracle.
5. Ghi utility trung bình/median, quantile, harm rate, net corrections, coverage thay support, cosine tới oracle gradient và overhead. Chấm cả entropy và supervised CE để thấy khi hai mục tiêu lệch nhau.
6. Correlation toàn candidate chỉ báo cáo trên exhaustive hoặc sample độc lập có sampling design rõ. Correlation trên best/worst screened phải gắn nhãn thiên lệch; so sánh policy utility trên cùng query vẫn phải tách khỏi correlation claim.
7. Nếu entropy gradient thất bại chủ yếu ở query dự đoán sai, thử một nhánh multi-view entropy giới hạn, cost-matched, trên development split rồi xác nhận lại ở dữ liệu mới. Không thêm nhiều heuristic đồng thời.

## Todo

- [ ] Chạy focused tests `pytest -q tests/test_oracle_support_utility.py`, kiểm chứng bằng model/data thật trên tập nhỏ.
- [ ] Lưu raw action, score, CE/accuracy trước-sau, source/config/stream fingerprints và báo cáo GO/REVISE/STOP.

## Success Criteria

GO khi policy không nhãn có lợi ích thực so với random/no-swap, lặp lại trên phần xác nhận đã khóa, không chỉ giảm entropy. Báo cáo cả nhóm query baseline đúng/sai và mức bất định. Kết quả chưa rõ thì thu thêm mẫu đã định trước; kết quả âm thì sửa query signal hoặc dừng mở rộng. Ngưỡng effect/harm và budget phải chốt trong phase 1, không tự đặt sau khi xem số.

## Risks and rollback

Confidence sai gây confirmation bias. Query tự nằm trong memory có thể tạo ưu thế self-cosine; ghi self-support membership và giữ policy giống baseline. Không coi gradient nonfinite/zero như tín hiệu có ích. Nếu probe thay output Ramen hoặc oracle labels lọt vào selector, dừng, sửa và rerun bằng evidence directory mới.
