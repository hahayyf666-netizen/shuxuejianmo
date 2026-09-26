"""Freeze the five Attachment 4 T0 inputs and the numerical decision rule."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "E题数据" / "附件4-可解释专项视频样本与特征文件"
INVENTORY = REPO / "阶段4.2_Q3_v1_preflight" / "stage_c2_provenance" / "results" / "pair_inventory_20.csv"
SAMPLE_IDS = ("02", "03", "07", "13", "16")
MIRROR_COMMIT = "ee52115996266573d2e3c2a3e04fd19e52af74c8"
MIRROR_ROOT = (
    "https://huggingface.co/datasets/reeha-parkar/cmu-mosei-comp-seq"
    f"/resolve/{MIRROR_COMMIT}/data"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with INVENTORY.open(newline="", encoding="utf-8-sig") as source:
        inventory = {row["sample_id"]: row for row in csv.DictReader(source)}
    rows = []
    for sample_id in SAMPLE_IDS:
        expected = inventory[sample_id]
        paths = {
            "aligned_pkl": DATA / "对齐版本" / f"{sample_id}.pkl",
            "unaligned_pkl": DATA / "未对齐版本" / f"{sample_id}.pkl",
            "aligned_mp4": DATA / "对齐版本" / "videos" / f"{sample_id}.mp4",
            "unaligned_mp4": DATA / "未对齐版本" / "videos" / f"{sample_id}.mp4",
        }
        actual = {key: sha256(path) for key, path in paths.items()}
        checks = {
            "aligned_pkl": expected["aligned_pkl_sha256"],
            "unaligned_pkl": expected["unaligned_pkl_sha256"],
            "aligned_mp4": expected["aligned_video_sha256"],
            "unaligned_mp4": expected["unaligned_video_sha256"],
        }
        if any(actual[key] != checks[key] for key in paths):
            raise RuntimeError(f"{sample_id}: source SHA differs from frozen C-2 inventory")
        rows.append({
            "sample_id": sample_id,
            "files": {key: {"path": str(path), "sha256": actual[key], "bytes": path.stat().st_size}
                      for key, path in paths.items()},
            "content_positions": int(expected["content_positions"]),
            "audio_unaligned_length": int(expected["audio_unaligned_length"]),
            "vision_unaligned_length": int(expected["vision_unaligned_length"]),
            "vision_rows_with_unique_source_match": int(expected["vision_rows_with_unique_source_match"]),
        })
    contract = {
        "status": "T0_INPUT_AND_RULE_FROZEN_BEFORE_CANDIDATE_VALUES",
        "python": sys.version,
        "platform": platform.platform(),
        "sample_ids": SAMPLE_IDS,
        "selection_rule": "02/03 normal, 07 long-truncated, 13/16 C-2 vision anomalies; no model output or labels",
        "source_c2_inventory_sha256": sha256(INVENTORY),
        "sources": rows,
        "candidate_source": {
            "kind": "unofficial mirror of CMU-MOSEI CSD; must match competition rows numerically",
            "mirror_commit": MIRROR_COMMIT,
            "timestamped_words_url": f"{MIRROR_ROOT}/CMU_MOSEI_TimestampedWords.csd",
            "covarep_url": f"{MIRROR_ROOT}/CMU_MOSEI_COVAREP.csd",
            "facet_url": f"{MIRROR_ROOT}/CMU_MOSEI_VisualFacet42.csd",
            "original_source_urls": {
                "covarep": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd",
                "facet": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualFacet42.csd",
            },
        },
        "strict_row_match_rule": {
            "same_feature_dimension_required": True,
            "coordinate_max_abs_error_at_most": 1e-4,
            "relative_l2_error_at_most": 1e-6,
            "unique_matching_candidate_row_required": True,
            "second_best_max_abs_error_at_least": 1e-3,
            "ordered_matches_required": True,
            "negative_controls": ["wrong_sample", "reversed_candidate_order"],
            "no_offset_search_or_dtw_or_learned_projection": True,
            "raw_time_requires_segment_offset_and_presentation_pts_check": True,
        },
        "stop_rule": "If candidate source cannot be read or first two normal samples have zero strict unique row matches, stop that modality; do not expand to 20.",
        "model_or_inference_run": False,
    }
    result = HERE / "results"
    result.mkdir(parents=True, exist_ok=True)
    path = result / "t0_frozen_contract.json"
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": contract["status"], "sample_ids": SAMPLE_IDS, "contract_path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
