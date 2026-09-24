from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_AUDIT = ROOT / "outputs" / "q1" / "diagnostics" / "full100_correspondence_audit"
DEFAULT_STAGE1 = ROOT / "outputs" / "q1" / "pilot_selection_metadata.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze deterministic Q1 branch-check samples before feature runs")
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--stage1-csv", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "q1" / "v1_delivery")
    args = parser.parse_args()
    audit = read(args.audit_root / "audit_100.csv")
    stage1 = read(args.stage1_csv)
    by_key = {row["sample_key"]: row for row in audit}
    stage_by_key = {row["sample_key"]: row for row in stage1}
    if len(by_key) != 100 or len(stage_by_key) != 100 or set(by_key) != set(stage_by_key):
        raise SystemExit("FAIL: full100 audit and Stage1 metadata do not agree")
    reasons: dict[str, list[str]] = {}

    def add(key: str, reason: str) -> None:
        reasons.setdefault(key, []).append(reason)

    add("-s9qJ7ATP7w$_$6", "confirmed_word_mapping_edit_list_control")
    add("-mJ2ud6oKI8$_$6", "confirmed_text_audio_mismatch_control")
    add("-s9qJ7ATP7w$_$0", "unverified_content_with_zero_duration_word_diagnostic_control")
    silent = [r for r in audit if r.get("audio_speech_valid") == "0"]
    if len(silent) != 2:
        raise SystemExit(f"FAIL: expected two existing confirmed-silence rows, found {len(silent)}")
    for row in silent:
        add(row["sample_key"], "digital_silence_control")
    noface = sorted(r["sample_key"] for r in audit if r.get("face_feature_valid") == "0")
    if not noface:
        raise SystemExit("FAIL: no full-frame face-feature-missing control exists")
    add(noface[0], "lexicographically_first_no_face_feature_control")
    uncertain = sorted(r["sample_key"] for r in audit if r.get("alignment_mode") == "UNCERTAIN_REVIEW")
    if not uncertain:
        raise SystemExit("FAIL: no uncertain-evidence route exists")
    add(next(key for key in uncertain if key not in reasons), "lexicographically_first_clip_only_uncertain_control")
    max_duration = max(float(row["ffprobe_format_duration_sec"]) for row in stage1)
    longest = min((row for row in stage1 if float(row["ffprobe_format_duration_sec"]) == max_duration), key=lambda row: row["sample_key"])
    add(longest["sample_key"], "maximum_Stage1_duration_proxy_boundary")
    max_words = max(int(row["word_count_proxy"]) for row in stage1)
    longest_text = min((row for row in stage1 if int(row["word_count_proxy"]) == max_words), key=lambda row: row["sample_key"])
    add(longest_text["sample_key"], "maximum_Stage1_word_count_proxy_boundary")

    rows = []
    for key in sorted(reasons):
        rows.append({
            "sample_key": key,
            "video_id": by_key[key]["video_id"],
            "clip_id": by_key[key]["clip_id"],
            "reasons": reasons[key],
            "official_text": by_key[key]["official_text"],
            "source_mp4_sha256": by_key[key]["source_mp4_sha256"],
            "duration_proxy_sec": float(stage_by_key[key]["ffprobe_format_duration_sec"]),
            "word_count_proxy": int(stage_by_key[key]["word_count_proxy"]),
            "full_audit_alignment_mode": by_key[key]["alignment_mode"],
            "full_audit_text_av_time_mapping_status": by_key[key]["text_av_time_mapping_status"],
        })
    out = args.output_root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "branch_check_samples.json").write_text(json.dumps({
        "selection_rule_version": "q1-branch-check-v1",
        "selected_count": len(rows),
        "deduplicated": True,
        "input_audit_sha256": digest(args.audit_root / "audit_100.csv"),
        "input_stage1_sha256": digest(args.stage1_csv),
        "rules": [
            "fixed 3 A0 pilot samples are controls only; no old pilot file is changed",
            "include both confirmed digital-silence samples",
            "lexicographically first sample with face_feature_valid=0",
            "lexicographically first UNCERTAIN_REVIEW sample not already selected",
            "maximum Stage1 ffprobe_format_duration_sec, tie by sample_key",
            "maximum Stage1 word_count_proxy, tie by sample_key",
            "union and de-duplicate by sample_key before running",
        ],
        "samples": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = out / "branch_check_samples.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_key", "video_id", "clip_id", "reasons", "source_mp4_sha256", "duration_proxy_sec", "word_count_proxy", "full_audit_alignment_mode", "full_audit_text_av_time_mapping_status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (";".join(row["reasons"]) if key == "reasons" else row[key]) for key in writer.fieldnames})
    print(f"PASS: selected {len(rows)} samples")
    for row in rows:
        print(row["sample_key"], ";".join(row["reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
