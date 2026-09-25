"""Independently verify published C-2 exact row matches and gate counts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--unaligned", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = read_csv(args.results / "aligned_to_unaligned_row_matches_20.csv")
    media = read_csv(args.results / "media_pts_20.csv")
    pair_summary = json.loads((args.results / "pairwise_source_summary.json").read_text(encoding="utf-8"))
    media_summary = json.loads((args.results / "media_pts_summary.json").read_text(encoding="utf-8"))
    if len(records) != 1128 or len(media) != 20:
        raise AssertionError("C-2 row count changed")
    inspected = 0
    for number in range(1, 21):
        sample_id = f"{number:02d}"
        with (args.aligned / f"{sample_id}.pkl").open("rb") as stream:
            aligned = pickle.load(stream)
        with (args.unaligned / f"{sample_id}.pkl").open("rb") as stream:
            unaligned = pickle.load(stream)
        for record in (r for r in records if r["sample_id"] == sample_id):
            modality = record["modality"]
            position = int(record["official_seq_index"])
            length = int(unaligned[f"{modality}_lengths"])
            if int(record["unaligned_valid_length"]) != length:
                raise AssertionError(f"{sample_id}/{modality}/{position}: length mismatch")
            target = np.asarray(aligned[modality])[position]
            source = np.asarray(unaligned[modality])[:length]
            actual = [index for index in range(length) if np.array_equal(source[index], target)]
            published = [int(value) for value in record["exact_unaligned_indices_zero_based"].split(";")
                         if value]
            if actual != published or len(actual) != int(record["exact_source_index_count"]):
                raise AssertionError(f"{sample_id}/{modality}/{position}: row match mismatch")
            if record["raw_time_or_pts_verified"] != "False":
                raise AssertionError("raw time was incorrectly promoted")
            inspected += 1
    audio = [r for r in records if r["modality"] == "audio"]
    vision = [r for r in records if r["modality"] == "vision"]
    counts = {"audio_unique": sum(int(r["exact_source_index_count"]) == 1 for r in audio),
              "vision_unique": sum(int(r["exact_source_index_count"]) == 1 for r in vision),
              "vision_unmatched": sum(int(r["exact_source_index_count"]) == 0 for r in vision)}
    if (counts != {"audio_unique": 564, "vision_unique": 523, "vision_unmatched": 41}
            or pair_summary["by_modality"]["audio"]["rows_with_unique_unaligned_index"] != 564
            or pair_summary["by_modality"]["vision"]["rows_with_unique_unaligned_index"] != 523):
        raise AssertionError("pair summary counts mismatch")
    if (sum(int(r["video_header_nb_frames"]) != int(r["video_decoded_frames_with_pts"])
            for r in media) != 16
            or media_summary["samples_with_video_header_count_different_from_decoded_pts_count"] != 16
            or any(r["audio_feature_row_to_raw_pts_verified"] != "False" or
                   r["vision_feature_row_to_raw_pts_verified"] != "False" for r in media)):
        raise AssertionError("media summary or raw support gate mismatch")
    receipt = {"status": "PASS", "published_rows_rechecked_against_original_pkls": inspected,
               **counts, "media_rows_checked": len(media),
               "video_header_vs_decoded_count_differences": 16,
               "raw_media_feature_row_time_verified": 0,
               "model_loaded": False, "training_run": False, "predictions_computed": False}
    (args.results / "c2_verification.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
