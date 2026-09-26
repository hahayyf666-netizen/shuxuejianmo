"""Reconcile C2/T2/T3 scope and recertify Q3 4.5 after the frozen test run.

Reads existing artifacts only; does not run the model, change thresholds, or
rewrite historical 4.5 results. New files live in final_4_5_review/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "final_4_5_review"
IDS = [f"{n:02d}" for n in range(1, 21)]
MODES = {"text": "verified_text", "audio": "index_only", "vision": "index_only"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a: float, b: float, atol: float = 1e-10) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lineage_path = STAGE / "t2_row_boundary_and_source_inventory_2026-09-26/results/aligned_position_lineage_20.csv"
    c2_path = STAGE / "stage_c2_provenance/results/aligned_to_unaligned_row_matches_20.csv"
    t2_path = STAGE / "t2_row_boundary_and_source_inventory_2026-09-26/results/t2_aggregate_gate.json"
    t3_path = STAGE / "t3_media_origin_2026-09-26/results_v3/t3_media_origin_records.json"
    t3_aggregate_path = STAGE / "t3_media_origin_2026-09-26/results_v3/t3_aggregate_gate.json"
    old_accept_path = HERE / "internal_acceptance.json"
    contract_path = HERE / "assets/attachment4_explanation_contract.json"
    attachment_summary_path = HERE / "results/formal_attachment4_summary.json"
    attachment_manifest_path = HERE / "results/output_manifest.json"
    test_root = STAGE / "formal_test_evaluation_2026-09-26"
    test_summary_path = test_root / "test_evaluation_summary.json"
    test_predictions_path = test_root / "test_predictions.csv"
    test_manifest_path = test_root / "output_manifest.json"

    lineage = rows(lineage_path)
    c2 = rows(c2_path)
    t2 = read_json(t2_path)
    t3 = read_json(t3_path)
    t3_aggregate = read_json(t3_aggregate_path)
    contract = read_json(contract_path)
    attachment = read_json(attachment_summary_path)
    old_accept = read_json(old_accept_path)
    test = read_json(test_summary_path)
    test_predictions = rows(test_predictions_path)

    by_sample: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in lineage:
        by_sample[item["sample_id"]][item["modality"]].append(item)
    t3_by_id = {item["sample_id"]: item for item in t3}
    assert len(lineage) == len(c2) == 1128
    assert set(by_sample) == set(IDS)
    assert len(t3_by_id) == len(t3) == 15

    mapping_rows = []
    for sample_id in IDS:
        a = by_sample[sample_id]["audio"]
        v = by_sample[sample_id]["vision"]
        assert len(a) == len(v) > 0
        c2_a = sum(x["c2_unique"] == "1" for x in a)
        c2_v = sum(x["c2_unique"] == "1" for x in v)
        t2_a = sum(x["source_row_status"] == "SOURCE_ROW_UNIQUE" for x in a)
        t2_v = sum(x["source_row_status"] == "SOURCE_ROW_UNIQUE" for x in v)
        native_nonunique_v = sum(x["source_row_status"] == "NATIVE_ROW_NONUNIQUE" for x in v)
        c2_unavailable_v = sum(x["source_row_status"] == "C2_CHAIN_UNAVAILABLE" for x in v)
        t3_item = t3_by_id.get(sample_id)
        t3_audio = "NOT_TESTED"
        t3_video = "NOT_TESTED"
        t3_media = "NOT_TESTED"
        if t3_item:
            t3_audio = (t3_item.get("audio") or {}).get("status") or "BLOCKED"
            t3_video = (t3_item.get("video") or {}).get("status") or "BLOCKED"
            t3_media = t3_item.get("media_identity") or "BLOCKED"
        mapping_rows.append({
            "sample_id": sample_id,
            "content_positions": len(a),
            "text_mapping_status": MODES["text"],
            "c2_audio_unique_count": c2_a,
            "c2_vision_unique_count": c2_v,
            "t2_audio_source_unique_count": t2_a,
            "t2_vision_source_unique_count": t2_v,
            "t2_vision_native_nonunique_count": native_nonunique_v,
            "t2_vision_c2_chain_unavailable_count": c2_unavailable_v,
            "t3_audio_origin_status": t3_audio,
            "t3_vision_origin_status": t3_video,
            "t3_media_identity_status": t3_media,
            "audio_formal_mapping_status": MODES["audio"],
            "vision_formal_mapping_status": MODES["vision"],
            "audio_raw_evidence_unavailable": 1,
            "vision_raw_evidence_unavailable": 1,
            "raw_evidence_unavailable": "audio|vision",
            "raw_evidence_reason": "official aligned audio/vision feature position to Attachment4 local media evidence is not contract-verified",
        })
    assert sum(int(x["t2_audio_source_unique_count"]) for x in mapping_rows) == 564
    assert sum(int(x["t2_vision_source_unique_count"]) for x in mapping_rows) == 522
    assert sum(int(x["t2_vision_native_nonunique_count"]) for x in mapping_rows) == 1
    assert sum(int(x["t2_vision_c2_chain_unavailable_count"]) for x in mapping_rows) == 41

    map_path = OUT / "c2_t2_t3_mapping_status_20.csv"
    with map_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    checks = []

    def check(name: str, condition: bool, evidence: str) -> None:
        checks.append({"name": name, "pass": bool(condition), "evidence": evidence})

    check("historical_4_5_gate", old_accept["status"] == "PASS_WITH_LIMITATIONS" and not old_accept["failed_checks"], "Original 36 checks remain in their historical receipt")
    check("c2_t2_row_counts", t2["aligned_position_count"] == 1128 and sum(int(r["content_positions"]) for r in mapping_rows) == 564, "1128 modality rows / 564 content positions")
    c2_key = lambda r: (r["sample_id"], r["modality"], int(r["official_seq_index"]))
    c2_by_key = {c2_key(r): r for r in c2}
    lineage_by_key = {c2_key(r): r for r in lineage}
    c2_join_ok = len(c2_by_key) == len(lineage_by_key) == len(c2) and set(c2_by_key) == set(lineage_by_key)
    if c2_join_ok:
        for key, source in c2_by_key.items():
            target = lineage_by_key[key]
            matches = source["exact_unaligned_indices_zero_based"].split(";") if source["exact_unaligned_indices_zero_based"] else []
            if target["c2_unique"] != ("1" if len(matches) == 1 else "0"):
                c2_join_ok = False
                break
            if len(matches) == 1 and target["unaligned_j"] != matches[0]:
                c2_join_ok = False
                break
    check("c2_t2_exact_key_join", c2_join_ok, "1128 unique sample/modality/index keys and exact C2 source matches")
    check("c2_t2_source_boundary", (t2["aligned_lineage"]["audio"]["SOURCE_ROW_UNIQUE"], t2["aligned_lineage"]["vision"]["SOURCE_ROW_UNIQUE"]) == (564, 522), "T2 audio 564 and vision 522 unique aligned source rows")
    check("known_vision_anomalies", {r["sample_id"] for r in mapping_rows if int(r["t2_vision_native_nonunique_count"])} == {"06"} and {r["sample_id"] for r in mapping_rows if int(r["t2_vision_c2_chain_unavailable_count"])} == {"13", "16"}, "06 nonunique; 13/16 broken C2 vision chain; 18 ambiguity outside aligned content is retained in T2 audit")
    full_origin_ids = {r["sample_id"] for r in mapping_rows if r["t3_media_identity_status"] == "PASS"}
    check("t3_origin_boundary", len(full_origin_ids) == t3_aggregate["source_media_pass_count"] == 9 and full_origin_ids == {"01", "02", "04", "07", "08", "14", "15", "17", "19"}, "9 complete media-origin samples; others do not inherit a pass")
    check("formal_explanation_boundary", all(r["text_mapping_status"] == "verified_text" and r["audio_formal_mapping_status"] == r["vision_formal_mapping_status"] == "index_only" and r["raw_evidence_unavailable"] == "audio|vision" for r in mapping_rows) and attachment["mapping_status"] == MODES == contract["prediction_contract"]["mapping_status"], "20 samples remain text verified; audio and vision index-only")
    check("test_model_identity", test["selected_architecture"] == "B0" and test["selected_checkpoint"] == "B0_seed2029.pt" and test["checkpoint_sha256"] == attachment["checkpoint_sha256"] and test["scaler_sha256"] == attachment["scaler_sha256"], "Test uses exact Attachment4 delivery model and train scaler hashes")
    check("test_split_isolation", test["evaluated_split"] == "test" and test["test_used_for_selection"] is False and test["model_or_threshold_changed"] is False and test["attachment4_results_modified"] is False, "One final test measurement; frozen selection and Attachment4 outputs unchanged")
    check("test_sample_identity", len(test_predictions) == test["sample_count"] == 727 and len({r["sample_id"] for r in test_predictions}) == 727, "727 unique test prediction rows")

    names = ["Negative", "Neutral", "Positive"]
    cm = [[0] * 3 for _ in range(3)]
    errors = []
    actual = []
    predicted = []
    for row in test_predictions:
        t = names.index(row["true_class"])
        p = names.index(row["predicted_class"])
        cm[t][p] += 1
        errors.append(float(row["predicted_intensity"]) - float(row["true_intensity"]))
        actual.append(float(row["true_intensity"]))
        predicted.append(float(row["predicted_intensity"]))
    accuracy = sum(cm[i][i] for i in range(3)) / 727
    f1 = []
    for i in range(3):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3) if j != i)
        fn = sum(cm[i][j] for j in range(3) if j != i)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    ax = sum(actual) / len(actual)
    px = sum(predicted) / len(predicted)
    cov = sum((a - ax) * (p - px) for a, p in zip(actual, predicted))
    va = sum((a - ax) ** 2 for a in actual)
    vp = sum((p - px) ** 2 for p in predicted)
    pearson = cov / math.sqrt(va * vp) if va > 0 and vp > 0 else None
    m = test["metrics"]
    numerical_ok = (
        cm == test["confusion_matrix"]["rows_true_cols_pred"]
        and close(accuracy, m["accuracy"])
        and close(sum(f1) / 3, m["macro_f1"])
        and all(close(f1[i], m["per_class_f1"][names[i]]) for i in range(3))
        and close(mae, m["mae"])
        and close(rmse, m["rmse"])
        and pearson is not None and close(pearson, m["pearson"])
    )
    check("test_metrics_independent_recalculation", numerical_ok, f"accuracy={accuracy:.6f}, macro_f1={sum(f1)/3:.6f}, mae={mae:.6f}, rmse={rmse:.6f}, pearson={pearson:.6f}")

    for label, root, manifest in (
        ("test", test_root, read_json(test_manifest_path)),
        ("attachment4", HERE / "results", read_json(attachment_manifest_path)),
    ):
        ok = all((root / rel).is_file() and sha256(root / rel) == expected for rel, expected in manifest.items())
        check(f"{label}_output_manifest", ok, f"{len(manifest)} output SHA entries verified")
    check("input_contract_hashes", test["aligned_pkl_sha256"] == "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd" and test["checkpoint_sha256"] == "723a9ddef831f35c25d95b325a51c194a8e7326460812ab5435c0b24fc8ad6ce", "Attachment2 and frozen B0 hashes match recorded contract")

    sample18_path = HERE / "results/samples/sample_18.json"
    predictions_csv_path = HERE / "results/predictions_explanations.csv"
    sample18 = read_json(sample18_path)
    original_note = "T2 vision row nonunique"
    sample18_lineage = by_sample["18"]["vision"]
    erratum_ok = (
        sample18["known_input_anomaly"] == original_note
        and len(sample18_lineage) == 48
        and all(r["source_row_status"] == "SOURCE_ROW_UNIQUE" for r in sample18_lineage)
        and read_json(attachment_manifest_path)["samples/sample_18.json"] == sha256(sample18_path)
        and read_json(attachment_manifest_path)["predictions_explanations.csv"] == sha256(predictions_csv_path)
    )
    check("sample18_annotation_erratum", erratum_ok, "Frozen annotation is corrected by a separately hashed, metadata-only erratum")
    erratum = {
        "status": "METADATA_ONLY_ERRATUM",
        "sample_id": "18",
        "field": "known_input_anomaly",
        "original_value": original_note,
        "corrected_interpretation": "One unaligned native vision row has nonunique source lineage; all 48 aligned vision content rows used by the model have unique source rows.",
        "original_artifacts": {
            "results/samples/sample_18.json": sha256(sample18_path),
            "results/predictions_explanations.csv": sha256(predictions_csv_path),
        },
        "formal_vision_mapping_status": "index_only",
        "formal_vision_mapping_reason": "Attachment4 local media time/frame mapping is not contract-verified",
        "unchanged": ["frozen selection contract", "predictions", "Shapley", "conditional IG", "output manifest"],
    }
    erratum_path = OUT / "metadata_erratum.json"
    erratum_path.write_text(json.dumps(erratum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "gate": "Q3_4_5_FINAL_REVIEW",
        "status": "PASS_WITH_LIMITATIONS" if all(c["pass"] for c in checks) else "FAIL",
        "checks": checks,
        "failed_checks": [c["name"] for c in checks if not c["pass"]],
        "test_evaluation": {"sample_count": 727, "accuracy": accuracy, "macro_f1": sum(f1) / 3, "mae": mae, "rmse": rmse, "pearson": pearson, "used_for_selection": False},
        "mapping": {"samples": 20, "content_positions": 564, "audio_source_unique": 564, "vision_source_unique": 522, "vision_nonunique": 1, "vision_c2_unavailable": 41, "complete_media_origin_samples": 9, "audio_vision_formal_mapping": "index_only"},
        "source_sha256": {p.relative_to(STAGE).as_posix(): sha256(p) for p in [lineage_path, c2_path, t2_path, t3_path, t3_aggregate_path, old_accept_path, contract_path, attachment_summary_path, attachment_manifest_path, test_summary_path, test_predictions_path, test_manifest_path]},
        "mapping_table_sha256": sha256(map_path),
        "metadata_erratum_sha256": sha256(erratum_path),
        "decision_boundary": "4.5 final review complete; 4.6 independent review and 4.7 freeze are separate gates",
    }
    (OUT / "q3_4_5_final_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# Q3 4.5 最终内部复核\n\n**状态：{result['status']}**。新增 test 评价后，独立重算727条逐样本结果：Accuracy={accuracy:.6f}，Macro-F1={sum(f1)/3:.6f}，MAE={mae:.6f}，RMSE={rmse:.6f}，Pearson={pearson:.6f}；全部与冻结测试摘要一致。\n\n## 解释证据边界\n\n`c2_t2_t3_mapping_status_20.csv` 汇总20条样本、564个内容位置：音频来源行564/564唯一，视觉来源行522/564唯一，1行原生来源二义，41行C2链不可用。T3有9条完整媒体起点通过。上述证据没有把官方模型输入行完整验证到附件4本地音频时段或视觉帧，因此正式状态继续为文本 `verified_text`，音频和视觉 `index_only`；逐样本明确 `raw_evidence_unavailable=1`。\n\n06的视觉来源行二义；13、16的视觉C2链断开；18的原生未对齐视觉行二义但不在模型aligned内容行。异常样本均保留在正式20条输出。\n\n## 测试与产物核验\n\n本次只读核对了冻结checkpoint、train scaler、输入数据SHA、727条唯一test样本、混淆矩阵、Attachment4和test输出清单。test未参与架构或阈值选择；附件4预测与解释工件未改变。共{len(checks)}项核查，失败{len(result['failed_checks'])}项。完整逐项证据见 `q3_4_5_final_review.json`。\n\n## 阶段结论\n\n4.5最终内部复核达到 `{result['status']}`。它允许把当前冻结产物提交4.6独立审核，但不构成独立审核通过或4.7最终封存。音频秒级证据、视觉关键帧、因果解释和“解释准确率”仍不属于当前正式交付范围。\n"""
    report = report.replace("raw_evidence_unavailable=1", "raw_evidence_unavailable=audio|vision（仅指音视频）")
    report += "\n## 冻结输出注释勘误\n\n正式结果里18号的 `known_input_anomaly=T2 vision row nonunique` 表述过宽。准确含义是：一条未对齐原生视觉来源行有二义，18号模型使用的48条 aligned 视觉内容行均有唯一来源行。该勘误见 `metadata_erratum.json`，其中记录原样本文件及预测 CSV 的 SHA；正式结果、合同和 manifest 保持原样。18号视觉正式解释仍为 `index_only`，因为附件4本地媒体时间或帧映射未通过合同验证。\n"
    (OUT / "Q3_4_5_FINAL_REVIEW_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": result["failed_checks"], "mapping_rows": len(mapping_rows)}, ensure_ascii=False))
    if result["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

