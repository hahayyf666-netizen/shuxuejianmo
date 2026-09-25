"""Attachment4 input/evidence precheck only: no model loading or prediction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np

from q3v1.data import sha256_file, validate_content_indices
from q3v1.mapping import index_only, raw_text_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected = ("01", "13", "20")  # fixed by input coverage, before any prediction
    rows = []
    samples = []
    for name in selected:
        pkl = args.aligned_directory / f"{name}.pkl"
        video = args.aligned_directory / "videos" / f"{name}.mp4"
        if not pkl.is_file() or not video.is_file():
            raise FileNotFoundError(f"missing paired Attachment4 input: {name}")
        with pkl.open("rb") as stream:
            data = pickle.load(stream)
        if str(data.get("id")) != name:
            raise ValueError(f"Attachment4 id mismatch: {name}")
        positions = validate_content_indices(data["text_bert"])
        for key, dim in (("text", 768), ("audio", 74), ("vision", 35)):
            feature = np.asarray(data[key])
            if feature.shape != (50, dim) or not np.isfinite(feature).all():
                raise ValueError(f"Attachment4 feature shape/nonfinite: {name}/{key}")
        media_sha = sha256_file(video)
        pkl_sha = sha256_file(pkl)
        source_hashes = {"text": raw_text_hash(str(data["raw_text"])),
                         "audio": media_sha, "vision": media_sha}
        for modality in ("text", "audio", "vision"):
            for position in positions:
                rows.append(index_only(name, modality, int(position), source_hashes[modality],
                                       "official feature row-to-raw evidence relation not verified").to_dict())
        samples.append({"sample_id": name, "aligned_pkl_sha256": pkl_sha,
                        "video_sha256": media_sha, "raw_text_sha256": source_hashes["text"],
                        "content_position_count": int(len(positions)),
                        "content_position_min": int(positions.min()), "content_position_max": int(positions.max()),
                        "vision_all_zero": bool(np.all(np.asarray(data["vision"]) == 0)),
                        "model_prediction_computed": False})
    output = args.out / "representative_evidence_mapping.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mapping_gate": "OPEN_HARD_STOP_FOR_RAW_AUDIO_AND_VISION_EVIDENCE",
        "selection_rule": "predeclared Attachment4 IDs 01,13,20; 13 covers known all-zero aligned vision",
        "samples": samples, "mapping_rows": len(rows),
        "status_counts": {"index_only": len(rows), "verified_text": 0, "verified_time": 0},
        "reason": "Neither an exact tokenizer revision with official text-row provenance nor official aligned audio/vision-row raw support has been verified. No seconds or keyframes were generated.",
        "formal_inference_run": False,
    }
    (args.out / "mapping_precheck.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mapping_gate": summary["mapping_gate"], "rows": len(rows), "samples": selected}))


if __name__ == "__main__":
    main()
