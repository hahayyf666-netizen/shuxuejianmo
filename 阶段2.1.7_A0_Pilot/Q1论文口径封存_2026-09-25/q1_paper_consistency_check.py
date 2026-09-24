from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
MANIFEST = PROJECT_ROOT / "outputs/q1/v1_delivery/manifest.csv"
DRAFT = HERE / "Q1论文正文建议稿.md"
SUMMARY = HERE / "Q1论文附表_100条汇总.csv"
RECEIPT = HERE / "Q1论文一致性检查回执.json"
TIME_FIG = PROJECT_ROOT / "outputs/q1/v1_delivery/examples/typical_correspondence_s9qJ7ATP7w_clip6.png"
HEAT_FIG = PROJECT_ROOT / "outputs/q1/paper_figures_2026-09-25/q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.png"
HEAT_MANIFEST = PROJECT_ROOT / "outputs/q1/paper_figures_2026-09-25/q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    data = pd.read_csv(MANIFEST)
    text = DRAFT.read_text(encoding="utf-8")
    heat = json.loads(HEAT_MANIFEST.read_text(encoding="utf-8"))

    mode_counts = Counter(data["alignment_mode"])
    mapping_counts = Counter(data["text_av_time_mapping_status"])
    expected_modes = {
        "TRI_MODAL_WORD_VALID": 5,
        "UNCERTAIN_REVIEW": 88,
        "TRI_MODAL_WITH_AUDIO_CONTENT_INVALID": 2,
        "AV_VALID_TEXT_UNALIGNED": 5,
    }
    expected_mapping = {"word_valid": 5, "clip_only": 90, "unavailable": 5}

    summary = data[
        [
            "sample_key",
            "modalities",
            "original_effective_duration_sec",
            "text_sequence_length",
            "audio_observation_length",
            "video_observation_length",
            "alignment_mode",
            "alignment_granularity",
            "text_av_time_mapping_status",
            "feature_dims_json",
            "output_file",
        ]
    ].copy()
    summary.to_csv(
        SUMMARY,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.6f",
    )

    forbidden_patterns = {
        "claims_all_100_word_aligned": r"100\s*条(?:样本)?(?:均|全部).{0,16}(?:词级|逐词).{0,8}(?:对齐|映射)",
        "claims_heatmap_emotion_intensity": r"(?:红色|蓝色|颜色).{0,12}(?:正向|负向|情感强度)",
        "claims_heatmap_modality_importance": r"颜色.{0,12}(?:模态贡献|模态重要性)",
    }
    forbidden_hits = {
        name: re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for name, pattern in forbidden_patterns.items()
    }
    # The required warning sentence contains the prohibited claim in quotation marks;
    # remove this exact negated occurrence before evaluating the first pattern.
    sanitized = text.replace("不能表述为“100条均完成词级三模态对齐”", "")
    forbidden_hits["claims_all_100_word_aligned"] = re.findall(
        forbidden_patterns["claims_all_100_word_aligned"], sanitized, flags=re.IGNORECASE | re.DOTALL
    )

    required_phrases = [
        "可信词级三模态对齐",
        "保守片段级关联",
        "无有效语音内容",
        "文本—A/V内容不一致",
        "UNCERTAIN_REVIEW",
        "典型可信word-level样本",
        "仅代表可信词级分支",
        "不表示情感强度、模态贡献或预测性能",
    ]
    checks = {
        "manifest_has_100_rows": len(data) == 100,
        "manifest_sample_keys_unique": data["sample_key"].nunique() == 100,
        "route_counts_match_5_88_2_5": dict(mode_counts) == expected_modes,
        "mapping_counts_match_5_90_5": dict(mapping_counts) == expected_mapping,
        "paper_contains_all_required_phrases": all(phrase in text for phrase in required_phrases),
        "paper_has_no_forbidden_positive_claim": all(not hits for hits in forbidden_hits.values()),
        "paper_numbers_match_manifest": all(
            token in text
            for token in [
                "1926个文本词单元",
                "77261个音频观测窗",
                "23241个视频帧",
                "71个有效词时间",
                "68个词位置具有有效词级视觉向量",
                "76条至少检测到一帧有效面部blendshape",
                "另外24条",
            ]
        ),
        "summary_has_100_rows": len(summary) == 100,
        "summary_keys_match_manifest": set(summary["sample_key"]) == set(data["sample_key"]),
        "time_figure_exists": TIME_FIG.is_file(),
        "heatmap_figure_exists": HEAT_FIG.is_file(),
        "heatmap_is_trusted_word_sample": heat["alignment_mode"] == "TRI_MODAL_WORD_VALID"
        and heat["text_av_time_mapping_status"] == "word_valid",
        "heatmap_checks_pass": bool(heat["all_checks_pass"]),
        "heatmap_dimensions_match_paper": heat["dimensions"]
        == {"text": [5, 768], "audio": [5, 50], "vision": [5, 104]},
    }
    receipt = {
        "scope": "Q1 manuscript wording, route counts, table and figure-caption consistency; no algorithm or feature modification",
        "manifest": MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST),
        "paper_draft": DRAFT.relative_to(PROJECT_ROOT).as_posix(),
        "paper_draft_sha256": sha256(DRAFT),
        "summary_table": SUMMARY.relative_to(PROJECT_ROOT).as_posix(),
        "summary_table_sha256": sha256(SUMMARY),
        "route_counts": dict(mode_counts),
        "mapping_counts": dict(mapping_counts),
        "forbidden_hits": forbidden_hits,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "decision": "Q1_PAPER_WORDING_FROZEN" if all(checks.values()) else "NEEDS_REVISION",
        "boundary": "The full competition manuscript does not yet exist; this receipt freezes the canonical Q1 section and its evidence, not Q2/Q3 or final-paper layout.",
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not receipt["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
