"""Summarize independently validated canonical evidence and author one report payload."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics as st
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "reports"
EVIDENCE = PROJECT / "evidence/kaggle-imports/canonical-hf-dc3cbf0"
METHODS = ["NoAdapt", "Ramen", "EntropyGatedRamen", "ConsensusRamen", "OracleDropOODRamen", "OracleIDGradientRamen", "OracleConsensusRamen"]
STREAMS = {"iid_mixed": "IID", "block": "Block", "recurring": "Recurring"}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation = json.loads((EVIDENCE / "validation.json").read_text())
    assert validation["status"] == "passed" and validation["validated_runs"] == 252
    rows = json.loads((EVIDENCE / "per-run-metrics.json").read_text())
    analysis = json.loads((EVIDENCE / "recomputed-analysis.json").read_text())
    cells = analysis["comparisons"]
    paired = []
    for c in cells:
        m = c["methods"]
        paired.append({"ood_ratio": c["ood_ratio"], "stream": c["stream_mode"], "seed": c["seed"],
                       "ramen_id_accuracy": m["Ramen"]["id_accuracy"], "consensus_id_accuracy": m["ConsensusRamen"]["id_accuracy"],
                       "delta_pp": 100 * c["consensus_vs_ramen_id_accuracy_gain"],
                       "auroc_delta": None if c["ood_ratio"] == 0 else m["ConsensusRamen"]["auroc"]-m["Ramen"]["auroc"],
                       "fpr95_delta_pp": None if c["ood_ratio"] == 0 else 100*(m["ConsensusRamen"]["fpr95"]-m["Ramen"]["fpr95"]),
                       "forward_cost_ratio": c["consensus_vs_ramen_cost_overhead"]["ConsensusRamen"]["forward_latency_total_ms_ratio"]})
    assert all(p["delta_pp"] < 0 for p in paired)
    aggregate = []
    for method in METHODS:
        ms = [r for r in rows if r["method"] == method]
        ood = [r for r in ms if r["ood_ratio"] > 0]
        result = {"method": method, "id_accuracy": st.mean(r["id_accuracy"] for r in ms),
                  "auroc": st.mean(r["auroc"] for r in ood), "fpr95": st.mean(r["fpr95"] for r in ood),
                  "h_score": st.mean(r["h_score"] for r in ood),
                  "negative_id_window_rate": st.mean(r["negative_id_window_rate"] for r in ms) if method != "NoAdapt" else None,
                  "forward_total_ms": st.mean(r["forward_total_ms"] for r in ms),
                  "peak_device_gib": st.mean(r["peak_device_memory_bytes"] for r in ms)/2**30,
                  "retained_memory_mib": (
                      st.mean(r["retained_memory_bytes"] for r in ms)/2**20
                      if all(r["retained_memory_bytes"] is not None for r in ms) else None
                  )}
        for ratio in [0, .1, .3, .5]:
            result[f"id_ood_{int(ratio*100)}"] = st.mean(r["id_accuracy"] for r in ms if r["ood_ratio"] == ratio)
        aggregate.append(result)
    by_method = {r["method"]: r for r in aggregate}
    seed_groups = []
    for ratio in [0, .1, .3, .5]:
        for stream in STREAMS:
            selected = [p for p in paired if p["ood_ratio"] == ratio and p["stream"] == stream]
            seed_groups.append({"OOD": f"{ratio:.0%}", "Stream": STREAMS[stream], "ood_ratio": ratio,
                                "delta_pp": st.mean(p["delta_pp"] for p in selected),
                                "seed_sd_pp": st.stdev(p["delta_pp"] for p in selected),
                                "minimum_pp": min(p["delta_pp"] for p in selected), "maximum_pp": max(p["delta_pp"] for p in selected),
                                "n_seeds": 3, "mean_ramen_id": st.mean(p["ramen_id_accuracy"] for p in selected),
                                "mean_consensus_id": st.mean(p["consensus_id_accuracy"] for p in selected)})

    # Independently compare evaluator sample identity and prediction changes on raw ZIP traces.
    diagnostics = []
    identity_fields = ("sample_idx", "ground_truth_domain", "original_label", "is_ood", "known_label_or_minus_one")
    run_index = {(r["ood_ratio"], r["stream_mode"], r["seed"], r["method"]): r for r in rows}
    with zipfile.ZipFile(validation["archive"]) as archive:
        def trace(run):
            with archive.open(f"nb-ramen-full-hf-dc3cbf0/canonical/{run['run_id']}/trace.jsonl") as handle:
                return [json.loads(line) for line in handle]
        for c in cells:
            key = (c["ood_ratio"], c["stream_mode"], c["seed"])
            baseline = trace(run_index[(*key, "NoAdapt")])
            for method in METHODS[1:]:
                current = trace(run_index[(*key, method)])
                assert len(current) == len(baseline) == 6000
                assert all(tuple(a[f] for f in identity_fields) == tuple(b[f] for f in identity_fields) for a,b in zip(current,baseline))
                id_pairs = [(a,b) for a,b in zip(current, baseline) if not b["is_ood"]]
                diagnostics.append({"ood_ratio": key[0], "stream": key[1], "seed": key[2], "method": method,
                                    "sample_identity_match": True,
                                    "pre_prediction_disagreements": sum(a["pre_adaptation_prediction"] != b["prediction"] for a,b in zip(current,baseline)),
                                    "id_fixed_vs_noadapt": sum(a["correct"] and not b["correct"] for a,b in id_pairs),
                                    "id_broken_vs_noadapt": sum(not a["correct"] and b["correct"] for a,b in id_pairs)})
    save("paired-diagnostics.json", diagnostics)
    save("validation.json", validation)
    save("summary-metrics.json", {"methods": aggregate, "paired": paired, "seed_groups": seed_groups})
    write_csv("per-run-metrics.csv", rows)
    write_csv("paired-comparisons.csv", paired)
    write_csv("seed-group-comparisons.csv", seed_groups)

    now = datetime.now(timezone.utc).isoformat()
    delta = st.mean(p["delta_pp"] for p in paired)
    cost = st.mean(p["forward_cost_ratio"] for p in paired)
    sources = [
        {"id": "metrics", "label": "Canonical CUDA metrics: 252 validated runs", "path": "per-run-metrics.csv",
         "query": {"engine": "python", "language": "python", "description": "Independent trace validation and equal-weight run aggregation; ID uses 36 cells/method, OOD detection excludes ratio zero and uses 27 cells/method."}},
        {"id": "paired", "label": "Paired Consensus minus Ramen comparisons", "path": "paired-comparisons.csv"},
        {"id": "validation", "label": "Strict source, plan and trace validation", "path": "validation.json"},
    ]
    blocks = []
    def markdown(identifier, text, source=None):
        block = {"id": identifier, "type": "markdown", "body": text}
        if source: block["sourceId"] = source
        blocks.append(block)
    def chart(identifier):
        blocks.append({"id": identifier+"-block", "type": "chart", "chartId": identifier, "layout": "full"})
    def table(identifier):
        blocks.append({"id": identifier+"-block", "type": "table", "tableId": identifier, "layout": "full"})

    title = "Đánh giá full matrix: nên đóng ConsensusRamen-v0 ở cấu hình hiện tại"
    markdown("title", "# " + title)
    markdown("summary", f"## Kết luận nghiên cứu\n\n**Đủ cơ sở dừng mở rộng ConsensusRamen-v0 hard-mask trong giao thức đã thử.** Consensus đạt **{by_method['ConsensusRamen']['id_accuracy']:.2%}** ID accuracy, Ramen đạt **{by_method['Ramen']['id_accuracy']:.2%}**; chênh lệch trung bình **{delta:.3f} điểm phần trăm**, âm ở **36/36** cấu hình ghép cặp. Đây là kết quả âm của cơ chế/cấu hình đã khóa, không bác bỏ toàn bộ hướng gradient compatibility hay QCGS.\n\nCó đánh đổi: FPR95 cải thiện nhẹ, nhưng chưa có bằng chứng đủ để đổi lấy suy giảm ID accuracy theo mục tiêu ban đầu là bảo vệ thích nghi hữu ích. Không đặt lại ngưỡng thành công sau khi nhìn kết quả.", "metrics")
    markdown("scope", "## Phép so sánh và cách đọc số liệu\n\n252 lượt = 7 phương pháp × 4 tỷ lệ OOD × 3 kiểu luồng × 3 seed. Mỗi lượt có 6.000 mẫu: 400 mẫu/domain × 15 corruption ở severity 5. Các file dữ liệu chứa đủ 19 corruption, nhưng benchmark chính dùng 15 corruption chuẩn. CLIP ViT-B/16; split v1 gồm 80 lớp đã biết và 20 lớp giữ ngoài; batch 100, block 64, hard-mask τ=0,2 và tối thiểu 3 lớp.\n\n**ID accuracy** là tỷ lệ dự đoán đúng trong các mẫu thuộc lớp đã biết. Bảng trung bình cho mỗi cấu hình trọng số bằng nhau; không gộp các mức OOD làm thay đổi mẫu số. Mỗi mức OOD lấy trung bình 9 lượt/phương pháp. Bảng seed dùng mean ± sample SD trên đúng 3 seed, không phải khoảng tin cậy. **OOD=0 không có AUROC/FPR95/H-score**, nên các trung bình OOD chỉ dùng 27 lượt/phương pháp.", "metrics")
    markdown("id-finding", "## Consensus giảm accuracy ở mọi kiểu luồng và mọi seed\n\nBiểu đồ là Consensus trừ Ramen theo từng tỷ lệ OOD và luồng, trung bình 3 seed. Mọi nhóm đều âm; ở từng seed, khoảng chênh lệch là **−1,907 đến −0,583 điểm phần trăm**. Ngay tại OOD=0, Consensus cũng thấp hơn: tác hại của hard-mask xuất hiện cả khi không có ảnh lạ. Điều này không ủng hộ mục tiêu giữ nguyên hiệu quả thích nghi trên dữ liệu ID.", "paired")
    chart("id-delta")
    table("accuracy-by-ood")
    markdown("seed-detail", "## Kết quả âm không do một seed đơn lẻ\n\nBảng dưới giữ riêng 12 nhóm OOD/luồng, mỗi nhóm 3 seed. Độ lệch chuẩn mô tả độ dao động giữa seed; 36 cấu hình chia sẻ dữ liệu và thiết kế nên không được xem là 36 phép thử độc lập để tuyên bố ý nghĩa thống kê phổ quát.", "paired")
    table("seed-detail")
    markdown("ood-finding", f"## Cải thiện OOD còn nhỏ và có đánh đổi\n\nTrên 27 cấu hình có OOD, AUROC của Consensus tăng từ **{by_method['Ramen']['auroc']:.4f} lên {by_method['ConsensusRamen']['auroc']:.4f}**, với 16 cấu hình tăng và 11 giảm. FPR95 giảm từ **{by_method['Ramen']['fpr95']:.2%} xuống {by_method['ConsensusRamen']['fpr95']:.2%}**, cải thiện ở 24/27 cấu hình, nhưng vẫn đánh nhầm khoảng 84% mẫu ID thành OOD tại ngưỡng phát hiện ít nhất 95% OOD.\n\nAUROC đo khả năng xếp hạng ảnh lạ; càng cao càng tốt. FPR95 đo tỷ lệ báo động nhầm trên ID ở ngưỡng recall OOD ≥95%; càng thấp càng tốt. **H-score trong repo kết hợp ID accuracy và OOD recall, không trực tiếp phạt FPR**, nên không dùng riêng H-score để khẳng định phát hiện OOD tốt.", "metrics")
    chart("ood-fpr")
    table("detection")
    markdown("mechanism", "## Các control chưa ủng hộ cơ chế hard-mask\n\n**EntropyGatedRamen cũng thấp hơn Ramen ở 36/36 cấu hình**, trung bình −2,807 điểm phần trăm. Gate giảm tỷ lệ OOD trong nhóm được nhận: từ 10% xuống 6,19%, từ 30% xuống 19,32%, từ 50% xuống 36,11%. Bộ nhớ sạch hơn theo tiêu chí này vẫn đi cùng accuracy thấp hơn.\n\n**OracleIDGradientRamen chỉ tăng trung bình 0,018 điểm phần trăm ID accuracy** so với Ramen trên 36 cấu hình; OracleDropOOD giảm 0,311 điểm. Cả hai cải thiện OOD detection rõ hơn: OracleID tăng AUROC 0,0470 và giảm FPR95 9,94 điểm trên 27 cấu hình có OOD. Trong benchmark này, ảnh hưởng của OOD thể hiện rõ hơn ở detection so với lợi ích ID accuracy của các oracle này. Oracle là control dùng nhãn thật, không phải upper bound được chứng minh hay phương pháp triển khai.\n\nConsensus giữ trung bình **53,13%** tọa độ gradient. Diagnostic của OracleID ở OOD>0 cho thấy mask loại khoảng **79,99%** tọa độ sai dấu, nhưng cũng loại **45,01%** tọa độ đúng dấu; chỉ số giảm hỏng hướng gradient trung bình là **−0,0666**, tức hướng sau mask tệ hơn theo diagnostic này. Đây là bằng chứng chẩn đoán về đánh đổi của mask, chưa chứng minh quan hệ nhân quả giữa từng tọa độ bị loại và từng lỗi dự đoán.", "metrics")
    markdown("cost", f"## Chi phí thấp thêm nhưng độ ổn định ID kém hơn\n\nConsensus tăng trung bình **{(cost-1)*100:.2f}%** thời gian forward trong từng cặp với Ramen. Tỷ lệ cửa sổ ID có accuracy thấp hơn NoAdapt tăng từ **{by_method['Ramen']['negative_id_window_rate']:.2%} lên {by_method['ConsensusRamen']['negative_id_window_rate']:.2%}**. Chi phí là đường xử lý đồng bộ gồm thích nghi và dự đoán, không phải thời gian riêng của phép consensus. Phép đo bộ nhớ CUDA peak khoảng 12,25 GiB cho cả hai; không có lợi ích giảm bộ nhớ đáng kể để bù cho accuracy.", "metrics")
    markdown("validation", "## Bằng chứng đã được kiểm tra lại\n\nZIP nguyên gốc được giữ nguyên. Kiểm tra CRC đủ 1.545 mục; đối chiếu đúng source commit dc3cbf0 và 137 file trong source inventory; tái tạo kế hoạch 252 lượt và config locks từ source độc lập. Tất cả **252 lượt qua strict validator**, gồm 1.512.000 dòng trace, đúng sample/stream pairing và giới hạn quyền dùng nhãn của Oracle. Báo cáo JSON tái tính khớp hoàn toàn bản Kaggle, CSV cũng khớp.\n\nĐể đọc bằng máy local, chỉ các prefix đường dẫn repo/evidence trong bản JSON dẫn xuất được đổi; các byte trace, giá trị đo và ZIP gốc giữ nguyên. Không chạy script từ ZIP. Dataset ảnh và CLIP cache không nằm trong ZIP: audit xác nhận provenance đã ghi, không hash lại các byte trên Kaggle hay chạy lại GPU.", "validation")
    markdown("limits", "## Có thể đóng v0; chưa thể đóng toàn bộ hướng nghiên cứu\n\nKết quả áp dụng cho split v1, 15 corruption severity 5, 3 seed, batch-atomic B=100/block=64 và cấu hình đã khóa. Mẫu hiện tại cùng batch được đưa vào bộ nhớ trước retrieval; đây không phải phép thử causal nghiêm ngặt. Split khác, ablation và DomainNet chưa được thực thi trong ZIP này.\n\nChênh lệch giữa các tỷ lệ OOD không tự nó là tác động nhân quả của OOD: tập mẫu ID được chọn cũng thay đổi. Không tuyên bố hard-mask thất bại trên mọi dataset, và không suy ra QCGS thất bại từ kết quả của cơ chế consensus giữa các lớp.")
    markdown("next", "## Bước tiếp theo nên chốt\n\n1. Ghi nhận matrix chính hoàn tất và kết quả âm của **ConsensusRamen-v0 hard-mask**; giữ nguyên ZIP, config và báo cáo.\n2. Dừng mở rộng nguyên xi v0 chỉ để tìm benchmark có lợi. Nếu tiếp tục cùng cơ chế, cần giả thuyết sửa đổi cụ thể, tách tập phát triển/đánh giá và khóa tiêu chí trước lần chạy mới. Các ablation đã lên kế hoạch có thể dùng để chẩn đoán, không dùng để chọn kết quả thuận lợi rồi trình bày như xác nhận độc lập.\n3. Nếu tiếp tục roadmap QCGS đã có, ưu tiên diagnostic nhỏ về utility của support với tín hiệu query không nhãn; kiểm soát bằng Ramen/random/no-swap. Đây là câu hỏi khác với consensus giữa các lớp, và cần bằng chứng riêng trước full matrix.\n4. Chưa đánh dấu luận văn hay toàn bộ hướng gradient compatibility hoàn tất; các yêu cầu split robustness, ablation và DomainNet còn mở.")
    markdown("questions", "## Câu hỏi còn đáng kiểm tra\n\n- Tín hiệu gradient của chính query có chọn được support hữu ích hơn độ gần đặc trưng không?\n- Lợi ích detection của oracle có thể đạt bằng tín hiệu không nhãn mà giữ được ID accuracy không?\n- Chi phí loại tọa độ đúng dấu có giải thích được thất bại hard-mask bằng một diagnostic kiểm soát riêng không?")

    accuracy_table = [{"method": m["method"], **{f"ood_{v}": f"{m[f'id_ood_{v}']:.2%}" for v in [0,10,30,50]}, "all": f"{m['id_accuracy']:.2%}"} for m in aggregate]
    detection_table = [{"method": m["method"], "auroc": round(m["auroc"],4), "fpr95": m["fpr95"], "h_score": round(m["h_score"],4)} for m in aggregate]
    seed_table = [{"OOD": g["OOD"], "Stream": g["Stream"], "delta": f"{g['delta_pp']:+.3f} ± {g['seed_sd_pp']:.3f}", "range": f"{g['minimum_pp']:+.3f} đến {g['maximum_pp']:+.3f}", "ood_ratio": g["ood_ratio"]} for g in seed_groups]
    def spec_table(identifier, title, dataset, columns, sort, source="metrics"):
        return {"id": identifier, "title": title, "dataset": dataset, "sourceId": source, "defaultSort": {"field": sort, "direction": "asc"}, "columns": columns}
    tables = [spec_table("accuracy-by-ood", "ID accuracy theo tỷ lệ OOD", "accuracy", [{"field":"method","label":"Phương pháp"}]+[{"field":f"ood_{v}","label":f"OOD {v}%"} for v in [0,10,30,50]]+[{"field":"all","label":"Trung bình"}], "method"),
              spec_table("seed-detail", "Chênh lệch accuracy và độ dao động qua seed", "seeds", [{"field":"OOD","label":"OOD"},{"field":"Stream","label":"Luồng"},{"field":"delta","label":"Mean ± SD (điểm %)"},{"field":"range","label":"Min đến max (điểm %)"}], "OOD", "paired"),
              spec_table("detection", "Chỉ số detection trên các cấu hình có OOD", "detection", [{"field":"method","label":"Phương pháp"},{"field":"auroc","label":"AUROC ↑","format":"number"},{"field":"fpr95","label":"FPR95 ↓","format":"percent"},{"field":"h_score","label":"H-score ↑","format":"number"}], "method")]
    charts = [
        {"id":"id-delta", "title":"Chênh lệch ID accuracy: ConsensusRamen − Ramen", "subtitle":"Trung bình 3 seed trong mỗi nhóm; đơn vị điểm phần trăm", "showDescription":True,
         "type":"bar", "dataset":"seed_groups", "sourceId":"paired", "layout":"full", "settings":{"groupMode":"grouped","showValues":True},
         "encodings":{"x":{"field":"OOD","type":"ordinal","label":"Tỷ lệ OOD"},"y":{"field":"delta_pp","type":"quantitative","label":"Chênh lệch (điểm %)"},"color":{"field":"Stream","type":"nominal","label":"Luồng"}}, "valueFormat":"number", "referenceLines":[{"axis":"y","value":0,"label":"Bằng Ramen","color":"neutral"}],
         "series":[{"field":"IID","label":"IID","color":"blue"},{"field":"Block","label":"Block","color":"orange"},{"field":"Recurring","label":"Recurring","color":"olive"}]},
        {"id":"ood-fpr", "title":"Tỷ lệ ID bị báo động nhầm tại mức phát hiện 95% OOD", "subtitle":"Trung bình 27 cấu hình có OOD; thấp hơn là tốt hơn", "showDescription":True,
         "type":"horizontalBar", "dataset":"detection", "sourceId":"metrics", "layout":"full", "valueFormat":"percent", "settings":{"orientation":"horizontal","showValues":True},
         "encodings":{"x":{"field":"method","type":"nominal","label":"Phương pháp"},"y":{"field":"fpr95","type":"quantitative","label":"FPR95","format":"percent"}}, "series":[{"field":"fpr95","label":"FPR95","color":"blue"}]},
    ]
    artifact = {"surface":"report", "manifest":{"version":1,"surface":"report","title":title,"generatedAt":now,"sources":sources,"blocks":blocks,"charts":charts,"tables":tables,"cards":[]},
                "snapshot":{"version":1,"generatedAt":now,"status":"ready","datasets":{"seed_groups":seed_groups,"accuracy":accuracy_table,"seeds":seed_table,"detection":detection_table}}, "sources":sources}
    save("artifact.json", artifact)
    report_text = "\n\n".join(b["body"] for b in blocks if b["type"] == "markdown")
    # This is the auditable narrative source, not a second reader delivery mode.
    (OUT / "findings.md").write_text(report_text + "\n")
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}}, "cells":[
        {"cell_type":"markdown","metadata":{},"source":["# Canonical CUDA results: reproducible checks\n", "The archive is immutable. Run audit-canonical-results.py with the documented archive/source/output paths before this notebook. All metrics are post-adaptation; OOD=0 detection metrics stay null.\n"]},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["import json, statistics as st\n",f"from pathlib import Path\nroot = Path({str(EVIDENCE)!r})\n","validation = json.loads((root / 'validation.json').read_text())\n","assert validation['status'] == 'passed' and validation['validated_runs'] == 252\n","rows = json.loads((root / 'per-run-metrics.json').read_text())\n","[(m, st.mean(r['id_accuracy'] for r in rows if r['method'] == m)) for m in sorted({r['method'] for r in rows})]\n"]},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["cells = json.loads((root / 'recomputed-analysis.json').read_text())['comparisons']\n","deltas = [100*c['consensus_vs_ramen_id_accuracy_gain'] for c in cells]\n","assert len(deltas) == 36 and all(d < 0 for d in deltas)\n","{'mean_delta_pp': st.mean(deltas), 'min_pp': min(deltas), 'max_pp': max(deltas)}\n"]}]}
    save("analysis.ipynb", notebook)
    print(json.dumps({"delta_pp":delta,"cost_ratio":cost,"paired_trace_checks":len(diagnostics),"max_pre_prediction_disagreements":max(d['pre_prediction_disagreements'] for d in diagnostics),"report":str(OUT/'artifact.json')}, indent=2))


if __name__ == "__main__":
    main()
