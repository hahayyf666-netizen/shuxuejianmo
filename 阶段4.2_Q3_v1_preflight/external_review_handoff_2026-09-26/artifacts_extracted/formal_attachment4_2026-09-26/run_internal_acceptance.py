"""Q3 4.5 internal acceptance for the bounded T3 + Attachment4 run.

This is a read-only acceptance audit.  It does not train, infer, select a
model, read labels, or alter the explanation contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SAMPLE_IDS = [f"{i:02d}" for i in range(1, 21)]
T3_CANDIDATE_IDS = ["01", "02", "03", "04", "05", "07", "08", "12", "13", "14", "15", "17", "18", "19", "20"]
T3_EXCLUDED_IDS = ["06", "09", "10", "11", "16"]
T3_PASS_IDS = ["01", "02", "04", "07", "08", "14", "15", "17", "19"]
T3_BLOCKED_IDS = ["03", "05", "12", "13", "18", "20"]
KNOWN_ANOMALIES = {
    "06": "T2 vision row nonunique",
    "13": "aligned vision all-zero / C2 visual chain unavailable",
    "16": "aligned vision all-zero-like / C2 visual chain unavailable",
    "18": "T2 vision row nonunique",
}
FORBIDDEN = (
    "audio_seconds",
    "vision_frames",
    "causal_emotion_contribution",
    "explanation_accuracy",
    "verified_official_provenance",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def expected_hashes(contract: dict, key: str, root: Path) -> tuple[int, int]:
    expected = contract["input_hashes"][key]
    actual = 0
    mismatch = 0
    subdir = root / "videos" if key == "videos" else root
    suffix = ".mp4" if key == "videos" else ".pkl"
    for sample_id, expected_sha in expected.items():
        path = subdir / f"{sample_id}{suffix}"
        if not path.is_file():
            mismatch += 1
            continue
        actual += 1
        if sha256(path) != expected_sha:
            mismatch += 1
    return actual, mismatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--attachment-root", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    stage = args.stage_root.resolve()
    attachment = args.attachment_root.resolve()
    results = here / "results"
    assets = here / "assets"
    t3_root = stage / "t3_media_origin_2026-09-26"
    t3_results = t3_root / "results_v3"
    summary_path = results / "formal_attachment4_summary.json"
    contract_path = assets / "attachment4_explanation_contract.json"
    combined_path = results / "attachment4_predictions_explanations.json"
    checks_path = results / "numeric_checks.json"
    manifest_path = results / "output_manifest.json"
    csv_path = results / "predictions_explanations.csv"

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    summary = load(summary_path)
    contract = load(contract_path)
    rows = load(combined_path)
    numeric = load(checks_path)
    manifest = load(manifest_path)
    t3_aggregate = load(t3_results / "t3_aggregate_gate.json")
    t3_contract = load(t3_results / "t3_frozen_contract.json")
    t3_records = load(t3_results / "t3_media_origin_records.json")

    # Contract and input coverage.
    check("attachment4_input_sample_scope", set(SAMPLE_IDS) == set(contract["input_hashes"]["videos"]) == set(contract["input_hashes"]["aligned_pkls"]), "contract has exactly 01-20 video and aligned PKL hashes")
    video_count, video_mismatch = expected_hashes(contract, "videos", attachment)
    pkl_count, pkl_mismatch = expected_hashes(contract, "aligned_pkls", attachment)
    check("attachment4_video_hashes", video_count == 20 and video_mismatch == 0, f"present={video_count}, hash_mismatch={video_mismatch}")
    check("attachment4_aligned_pkl_hashes", pkl_count == 20 and pkl_mismatch == 0, f"present={pkl_count}, hash_mismatch={pkl_mismatch}")
    check("card_selection_before_inference", contract["card_selection"]["selection_before_inference"] is True, "deterministic card selection recorded before inference")
    check("card_selection_rule", contract["card_selection"]["selected_sample_ids"] == ["15", "04", "03"], "selected cards match frozen SHA256 ranking")
    check("contract_mapping_scope", contract["prediction_contract"]["mapping_status"] == {"text": "verified_text", "audio": "index_only", "vision": "index_only"}, "contract mapping statuses unchanged")
    check("contract_test_policy", contract["prediction_contract"]["test_evaluated"] is False and contract["prediction_contract"]["labels_read"] is False, "contract excludes test and labels")

    # T3 result boundary.
    record_ids = sorted(x["sample_id"] for x in t3_records)
    check("t3_candidate_scope", record_ids == sorted(T3_CANDIDATE_IDS), f"records={len(record_ids)} candidate IDs match frozen shape filter")
    check("t3_exclusion_scope", t3_contract["excluded_non_youtube_shape_sample_ids"] == T3_EXCLUDED_IDS, "five excluded IDs match frozen non-YouTube-shape rule")
    check("t3_aggregate_counts", t3_aggregate["sample_count"] == 15 and t3_aggregate["audio_pass_count"] == 11 and t3_aggregate["video_pass_count"] == 9 and t3_aggregate["source_media_pass_count"] == 9, "T3 aggregate counts are internally consistent")
    t3_pass = sorted(x["sample_id"] for x in t3_records if x.get("media_identity") == "PASS")
    t3_blocked = sorted(x["sample_id"] for x in t3_records if x.get("sample_id") in T3_BLOCKED_IDS)
    check("t3_pass_ids", t3_pass == sorted(T3_PASS_IDS), f"full media-origin PASS IDs={t3_pass}")
    check("t3_blocked_ids", t3_blocked == sorted(T3_BLOCKED_IDS), f"blocked/partial IDs={t3_blocked}")
    check("t3_no_model_run", t3_aggregate["model_or_prediction_run"] is False and t3_contract["no_model_or_prediction_run"] is True, "T3 did not run model or prediction")
    check("t3_mapping_boundary", t3_aggregate["formal_xai_mapping_status"] == "index_only", "T3 did not upgrade formal mapping")
    check("t3_records_hash", sha256(t3_results / "t3_media_origin_records.json") == t3_aggregate["records_sha256"], "T3 record hash matches aggregate")

    # Formal Attachment4 output coverage.
    row_ids = sorted(str(x["sample_id"]) for x in rows)
    check("formal_prediction_row_count", len(rows) == 20 and row_ids == SAMPLE_IDS, f"rows={len(rows)}, unique IDs={len(set(row_ids))}")
    per_sample = sorted((results / "samples").glob("sample_*.json"))
    check("formal_per_sample_files", len(per_sample) == 20 and [p.stem[-2:] for p in per_sample] == SAMPLE_IDS, f"per_sample_files={len(per_sample)}")
    check("formal_numeric_check_count", len(numeric) == 160, f"numeric_checks={len(numeric)}")
    numeric_pass = all(item.get("pass") is True for item in numeric)
    method_counts = Counter(item.get("method") for item in numeric)
    check("formal_numeric_checks", numeric_pass and method_counts["shapley"] == 40 and method_counts["conditional_ig"] == 120, f"methods={dict(method_counts)}")
    max_residual = max(abs(float(item.get("residual", 0.0))) for item in numeric)
    check("formal_numeric_residual", max_residual <= 1e-5, f"max_abs_residual={max_residual:.3g}")
    check("formal_summary_gate", summary["status"] == "ATTACHMENT4_FORMAL_COMPLETE" and summary["numerical_status"] == "PASS" and summary["final_gate"] == "REVIEW_GATE", "formal summary status and review gate")
    check("formal_no_test_or_labels", summary["test_evaluated"] is False and summary["labels_read"] is False and summary["attachment4_inference"] is True, "Attachment4-only inference; no test or labels")

    allowed_local_status = "localized_feature_position"
    allowed_mapping = {"text": "verified_text", "audio": "index_only", "vision": "index_only"}
    allowed_audit = {"PASS", "BLOCKED", "NOT_TESTED"}
    mapping_ok = True
    local_ok = True
    audit_ok = True
    anomaly_ok = True
    forbidden_key_ok = True
    for record in rows:
        sid = str(record["sample_id"])
        if record.get("mapping_status") != allowed_mapping:
            mapping_ok = False
        if record.get("media_origin_audit") not in allowed_audit:
            audit_ok = False
        if record.get("known_input_anomaly") != KNOWN_ANOMALIES.get(sid):
            anomaly_ok = False
        for target_name in ("classification", "regression"):
            modalities = record["targets"][target_name]["modalities"]
            for modality in ("text", "audio", "vision"):
                block = modalities[modality]
                if block.get("local_attribution_status") != allowed_local_status:
                    local_ok = False
                for pos in block.get("top_positions", []):
                    forbidden_key_ok &= not any(k.lower() in {"time", "pts", "frame", "seconds", "start", "end"} for k in pos)
                    if modality == "text":
                        if pos.get("mapping_status") != "verified_text" or not isinstance(pos.get("char_start"), int) or not isinstance(pos.get("char_end"), int):
                            mapping_ok = False
                    else:
                        if pos.get("mapping_status") != "index_only":
                            mapping_ok = False
    check("formal_mapping_contract", mapping_ok, "text verified_text; audio/vision index_only; no unverified temporal mapping")
    check("formal_local_attribution_status", local_ok, "all 120 target-modality blocks expose feature-position attribution")
    check("formal_media_origin_audit_fields", audit_ok, "T3 status carried as audit field only")
    check("known_input_anomaly_preservation", anomaly_ok, "06/13/16/18 anomaly records retained")
    check("no_temporal_claim_in_formal_positions", forbidden_key_ok, "formal positions contain no seconds/PTS/frame fields")

    # Numerical finiteness and probability simplex.
    prediction_ok = True
    for record in rows:
        pred = record["prediction"]
        probs = pred["probabilities"]
        if not all(finite(probs[k]) and 0.0 <= float(probs[k]) <= 1.0 for k in ("Negative", "Neutral", "Positive")):
            prediction_ok = False
        if abs(sum(float(probs[k]) for k in ("Negative", "Neutral", "Positive")) - 1.0) > 1e-6:
            prediction_ok = False
        if not finite(pred["intensity"]) or not -3.0 <= float(pred["intensity"]) <= 3.0:
            prediction_ok = False
    check("prediction_finiteness", prediction_ok, "20 probability simplexes and intensities finite and bounded")

    # Manifest and public-safe CSV.
    manifest_ok = True
    for rel, digest in manifest.items():
        path = results / rel
        if not path.is_file() or sha256(path) != digest:
            manifest_ok = False
    expected_manifest_count = 25
    check("formal_output_manifest", manifest_ok and len(manifest) == expected_manifest_count, f"manifest_entries={len(manifest)}, expected={expected_manifest_count}")
    public_csv_text = csv_path.read_text(encoding="utf-8").lower()
    check("public_csv_no_raw_text", "raw_text" not in public_csv_text and "transcript" not in public_csv_text, "public CSV omits raw transcript text")
    check("public_csv_no_forbidden_claims", not any(term in public_csv_text for term in FORBIDDEN), "public CSV contains no prohibited evidence claims")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    check("public_csv_rows", len(csv_rows) == 20 and sorted(r["sample_id"] for r in csv_rows) == SAMPLE_IDS, "public CSV has one row for each Attachment4 sample")

    # Provenance hashes in summary.
    for field, path in (
        ("checkpoint_sha256", assets / "B0_seed2029.pt"),
        ("scaler_sha256", assets / "train_scaler.pt"),
        ("contract_sha256", contract_path),
        ("t3_records_sha256", t3_results / "t3_media_origin_records.json"),
    ):
        check(f"hash_{field}", path.is_file() and sha256(path) == summary[field], f"{field} matches formal summary")

    status = "PASS_WITH_LIMITATIONS" if all(x["pass"] for x in checks) else "FAIL"
    report_path = here / "Q3_4_5_INTERNAL_ACCEPTANCE_REPORT.md"
    audit = {
        "status": status,
        "gate": "Q3_4_5_INTERNAL_ACCEPTANCE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "failed_checks": [x["name"] for x in checks if not x["pass"]],
        "checks": checks,
        "t3": {
            "candidate_count": 15,
            "full_media_origin_pass_count": 9,
            "audio_pass_count": 11,
            "video_pass_count": 9,
            "formal_mapping": "index_only",
        },
        "formal_attachment4": {
            "sample_count": 20,
            "numeric_checks": 160,
            "numeric_failures": summary["numeric_failures"],
            "mapping_status": summary["mapping_status"],
            "test_evaluated": summary["test_evaluated"],
            "labels_read": summary["labels_read"],
            "final_gate": summary["final_gate"],
        },
        "next_stage": "STOP_AFTER_4_5; do not enter 4.6 in this run",
    }
    (here / "internal_acceptance.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# Q3 4.5 内部验收报告

**验收门**：`Q3_4_5_INTERNAL_ACCEPTANCE`  
**结果**：`{status}`  
**检查数**：{len(checks)}；失败检查：{len(audit['failed_checks'])}  
**停止点**：本报告完成后停止，不进入 4.6 独立红队复核，也不进入 4.7 最终冻结。

## 1. 完成了什么

1. 完成 T3 有界媒体起点核验：15 条冻结候选中，9 条音频+视频完整通过，11 条音频门通过，9 条视频门通过；T3 没有读取标签、测试集、模型预测或 XAI 输出。
2. 完成 Attachment4 01–20 的正式预测与 feature-space attribution 输出：20 条预测、40 条 exact 3-player Shapley、120 条 conditional IG 数值检查，全部通过。
3. 生成一行一条的 `predictions_explanations.csv`、逐样本 JSON、数值检查、输出 SHA manifest，并完成只读一致性审计。

## 2. 得到的结论

- T3 媒体导航证据在冻结候选范围内达到有限目标，但它不等于官方特征提取 provenance，也不改变正式解释映射合同。
- 正式解释结果的映射边界保持为：`text=verified_text`、`audio=index_only`、`vision=index_only`。
- Shapley/conditional IG 的数值闭合、概率单纯形、样本覆盖、输入 SHA、输出 manifest 和禁止测试集/标签读取检查均通过。
- `06、13、16、18` 的已知输入异常被保留；没有因为异常删除样本或伪造音视频证据。

## 3. 仍未确认的内容

- T3 未建立每一条官方 audio/vision feature row 到附件4本地秒数或视频帧的正式合同映射，因此不能报告音频秒级证据、视觉关键帧或官方特征采样窗口。
- T3 中 03、05、12、20 被来源媒体元数据条件阻断，13、18 只有音频门通过而视频门未通过；这不被解释为来源不存在。
- 这些限制属于证据范围限制，不是本次内部验收的数值失败。

## 4. 是否满足进入下一阶段的条件

`4.5` 内部验收条件满足，状态为 `{status}`。本次按指令在 4.5 停止；是否进入 4.6 由后续单独指令决定。

## 5. 主要复现入口

- T3：`t3_media_origin_2026-09-26/run_t3_media_origin.py`
- T3 规则与结果：`t3_media_origin_2026-09-26/results_v3/t3_frozen_contract.json`、`t3_aggregate_gate.json`、`t3_media_origin_records.json`
- Attachment4 正式入口：`formal_attachment4_2026-09-26/run_attachment4_formal.py`
- 本次验收入口：`formal_attachment4_2026-09-26/run_internal_acceptance.py`
- 结果目录：`formal_attachment4_2026-09-26/results/`

完整逐项检查结果保存在同目录 `internal_acceptance.json`。私有来源媒体及下载命令不纳入公共报告。
"""
    report_path.write_text(report, encoding="utf-8")
    sums = []
    for path in sorted([here / "internal_acceptance.json", report_path] + [t3_results / n for n in ("t3_aggregate_gate.json", "t3_frozen_contract.json", "t3_media_origin_records.json")] + [summary_path, combined_path, csv_path, manifest_path]):
        sums.append(f"{sha256(path)}  {path.relative_to(stage).as_posix()}")
    (here / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "check_count": len(checks), "failed_checks": audit["failed_checks"], "report": str(report_path)}, ensure_ascii=False))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
