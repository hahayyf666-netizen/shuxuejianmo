"""Validate C-4 artifacts and summarize strict recovery separately from proxy similarity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pair-inventory", type=Path, required=True)
    args = parser.parse_args()
    base = args.results
    text = json.loads((base / "text_reconstruction.json").read_text(encoding="utf-8"))
    media = json.loads((base / "media_reconstruction.json").read_text(encoding="utf-8"))
    controls = json.loads((base / "similarity_controls.json").read_text(encoding="utf-8"))
    with args.pair_inventory.open(newline="", encoding="utf-8-sig") as stream:
        inventory = {row["sample_id"]: row for row in csv.DictReader(stream)}
    with (base / "text_row_trace_564.csv").open(newline="", encoding="utf-8-sig") as stream:
        trace = list(csv.DictReader(stream))
    expected = {f"{number:02d}" for number in range(1, 21)}
    text_by_id = {row["sample_id"]: row for row in text["sample_metrics"]}
    media_by_id = {row["sample_id"]: row for row in media["per_sample"]}
    control_by_id = {row["sample_id"]: row for row in controls["per_sample"]}
    if not all(set(mapping) == expected for mapping in
               (inventory, text_by_id, media_by_id, control_by_id)):
        raise ValueError("C-4 sample ID set is not 01–20 in all artifacts")
    if len(trace) != 564 or not all(row["strict_verified_within_1e-4"] == "True"
                                     for row in trace):
        raise ValueError("text row trace is incomplete or contains an unverified row")
    if text["text_row_trace_csv_sha256"] != sha256(base / "text_row_trace_564.csv"):
        raise ValueError("text row trace hash mismatch")
    if (text["total_content_rows"] != 564
            or text["total_rows_all_dims_within_1e-4"] != 564
            or text["total_all_50_positions_within_1e-4"] != 1000):
        raise ValueError("text reconstruction totals differ from expected content/50-position counts")
    table = []
    for sid in sorted(expected):
        tx, md, ctl = text_by_id[sid], media_by_id[sid], control_by_id[sid]
        if tx["pkl_sha256"] != inventory[sid]["aligned_pkl_sha256"]:
            raise ValueError(f"{sid}: text PKL hash differs from C-2")
        if (md["pkl_sha256"] != inventory[sid]["unaligned_pkl_sha256"]
                or md["video_sha256"] != inventory[sid]["unaligned_video_sha256"]):
            raise ValueError(f"{sid}: media input hash differs from C-2")
        if md["candidate_npz_sha256"] != sha256(base / "candidate_npz" / f"{sid}.npz"):
            raise ValueError(f"{sid}: candidate NPZ hash mismatch")
        if "error" in md["audio"] or "error" in md["vision"]:
            raise ValueError(f"{sid}: media extraction error present")
        table.append({
            "sample_id": sid,
            "text_content_rows": tx["content_rows"],
            "text_strict_rows": tx["rows_all_dims_within_1e-4"],
            "text_mean_cosine": tx["mean_row_cosine"],
            "text_mean_absolute_error": tx["mean_absolute_error"],
            "text_max_absolute_error": tx["max_absolute_error"],
            "audio_official_rows": md["official_audio_rows"],
            "audio_strict_74D_rows": md["audio"]["strict_full_74D_verified_rows"],
            "audio_proxy_CKA_20hz": md["audio"]["linear_CKA_fixed_20hz"],
            "audio_proxy_CKA_reversed": ctl["audio"]["reversed_CKA"],
            "vision_official_rows": md["official_vision_rows"],
            "vision_strict_35D_rows": md["vision"]["strict_full_35D_verified_rows"],
            "vision_proxy_face_valid_rows": md["vision"]["proxy_15hz_face_valid_rows"],
            "vision_proxy_CKA": md["vision"]["linear_CKA_on_face_valid_rows"],
            "vision_proxy_CKA_reversed": ctl["vision"]["reversed_CKA"],
        })
    if (sum(row["audio_official_rows"] for row in table) != 3677
            or sum(row["vision_official_rows"] for row in table) != 2593):
        raise ValueError("official media row totals changed")
    if any(row["audio_strict_74D_rows"] != 0 or row["vision_strict_35D_rows"] != 0
           for row in table):
        raise ValueError("a proxy row was incorrectly promoted to strict official identity")
    csv_path = base / "reconstruction_summary_20.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "transformers", "opensmile", "mediapipe", "av")}
    gate = {
        "scope": "Attachment 4 sample 01–20 only",
        "text": {"strict_recovered_content_rows": 564, "official_content_rows": 564,
                 "all_50_positions_reconstructed": 1000,
                 "mapping_status_for_tested_samples": "verified_text",
                 "similarity_metric": "full 768D content-row cosine and absolute error",
                 "maximum_absolute_error": max(row["text_max_absolute_error"] for row in table)},
        "audio": {"strict_recovered_74D_rows": 0, "official_unaligned_rows": 3677,
                  "mapping_status": "index_only",
                  "diagnostic_similarity_metric": "linear CKA, candidate 25D eGeMAPS versus official 74D",
                  "diagnostic_median_CKA": controls["summary"]["audio"]["median_observed_CKA"],
                  "reversed_order_median_CKA": controls["summary"]["audio"]["median_reversed_CKA"],
                  "failure_reason": "official 74D extractor and row time support not recovered"},
        "vision": {"strict_recovered_35D_rows": 0, "official_unaligned_rows": 2593,
                   "mapping_status": "index_only",
                   "diagnostic_similarity_metric": "linear CKA, candidate 52D MediaPipe versus official 35D",
                   "diagnostic_comparable_samples": controls["summary"]["vision"]["comparable_samples"],
                   "diagnostic_median_CKA": controls["summary"]["vision"]["median_observed_CKA"],
                   "reversed_order_median_CKA": controls["summary"]["vision"]["median_reversed_CKA"],
                   "failure_reason": "official 35D extractor and frame/PTS support not recovered"},
        "technical_fallback_scope": "feature-space attribution only; no raw-media evidence claims",
        "q3_protocol_modified": False,
        "existing_raw_evidence_training_gate": "UNCHANGED_STOP",
        "training_run": False,
        "q3_predictor_loaded": False,
        "environment": {"sys_executable": sys.executable, "python_version": sys.version,
                        "package_versions": versions},
        "source_pair_inventory_sha256": sha256(args.pair_inventory),
        "summary_csv_sha256": sha256(csv_path),
    }
    (base / "c4_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps({"text": gate["text"], "audio": gate["audio"],
                      "vision": gate["vision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
