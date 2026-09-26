"""Freeze Attachment4 explanation-card selection before model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
VIDEO_DIR = PROJECT / "E题数据" / "附件4-可解释专项视频样本与特征文件" / "对齐版本" / "videos"
PKL_DIR = PROJECT / "E题数据" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ids = [f"{i:02d}" for i in range(1, 21)]
    known_input_anomalies = {
        "06": "T2 vision row nonunique",
        "13": "aligned vision all-zero / C2 visual chain unavailable",
        "16": "aligned vision all-zero-like / C2 visual chain unavailable",
        "18": "T2 vision row nonunique",
    }
    normal = [sid for sid in ids if sid not in known_input_anomalies]
    ranked = sorted(normal, key=lambda sid: hashlib.sha256(("q3-card-v1|" + sid).encode()).hexdigest())
    cards = ranked[:3]
    contract = {
        "version": "q3-attachment4-formal-v1-2026-09-26",
        "scope": "all Attachment4 aligned 01-20 formal prediction and feature-space attribution",
        "model": "B0_seed2029",
        "card_selection": {
            "count": 3,
            "rule": "exclude only pre-audited input anomalies, then ascending SHA256(q3-card-v1|sample_id)",
            "known_input_anomalies": known_input_anomalies,
            "eligible_sample_ids": normal,
            "ranked_eligible_sample_ids": ranked,
            "selected_sample_ids": cards,
            "selection_before_inference": True,
        },
        "prediction_contract": {
            "feature_version": "aligned_50",
            "text_representation": "precomputed text only",
            "class_order": ["Negative", "Neutral", "Positive"],
            "mapping_status": {"text": "verified_text", "audio": "index_only", "vision": "index_only"},
            "allowed_explanation": "modality-dependent feature-space attribution",
            "local_top_k": 5,
            "targets": ["classification", "regression"],
            "prohibited_claims": ["audio_seconds", "vision_frames", "unverified_channel_physical_semantics", "causal_emotion_contribution", "explanation_accuracy"],
            "test_evaluated": False,
            "labels_read": False,
        },
        "input_hashes": {
            "videos": {sid: sha256(VIDEO_DIR / f"{sid}.mp4") for sid in ids},
            "aligned_pkls": {sid: sha256(PKL_DIR / f"{sid}.pkl") for sid in ids},
        },
    }
    out = HERE / "assets" / "attachment4_explanation_contract.json"
    out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_sample_ids": cards, "eligible_count": len(normal), "contract": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
