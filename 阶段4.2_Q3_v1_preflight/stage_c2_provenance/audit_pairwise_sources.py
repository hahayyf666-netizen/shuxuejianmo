"""Audit exact within-attachment feature lineage; never infer raw-media time."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_c1_evidence_mapping"))
from probe_tokenizer_mapping import validate_text_bert


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected one-sample PKL dictionary: {path}")
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--unaligned", type=Path, required=True)
    parser.add_argument("--problem-docx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    row_matches: list[dict] = []
    for number in range(1, 21):
        sample_id = f"{number:02d}"
        ap = args.aligned / f"{sample_id}.pkl"
        up = args.unaligned / f"{sample_id}.pkl"
        av = args.aligned / "videos" / f"{sample_id}.mp4"
        uv = args.unaligned / "videos" / f"{sample_id}.mp4"
        aligned = load(ap)
        unaligned = load(up)
        if str(aligned.get("id")) != sample_id or str(unaligned.get("id")) != sample_id:
            raise ValueError(f"{sample_id}: PKL identity mismatch")
        atb = validate_text_bert(aligned["text_bert"], sample_id=sample_id, vocab_size=30522)
        utb = validate_text_bert(unaligned["text_bert"], sample_id=sample_id, vocab_size=30522)
        content = np.flatnonzero(atb[1] == 1)[1:-1]
        text_equal = np.array_equal(aligned["text"], unaligned["text"])
        ids_equal = np.array_equal(atb, utb)
        raw_text_equal = aligned["raw_text"] == unaligned["raw_text"]
        hashes = {"aligned_pkl_sha256": sha256(ap), "unaligned_pkl_sha256": sha256(up),
                  "aligned_video_sha256": sha256(av), "unaligned_video_sha256": sha256(uv)}
        info = {"sample_id": sample_id, **hashes,
                "video_bytes_equal": hashes["aligned_video_sha256"] == hashes["unaligned_video_sha256"],
                "id_equal": True, "raw_text_equal": raw_text_equal,
                "text_50x768_equal": text_equal, "text_bert_3x50_equal": ids_equal,
                "aligned_pkl_keys": ";".join(sorted(aligned)),
                "unaligned_pkl_keys": ";".join(sorted(unaligned)),
                "text_dtype_aligned": str(np.asarray(aligned["text"]).dtype),
                "text_dtype_unaligned": str(np.asarray(unaligned["text"]).dtype),
                "content_positions": len(content),
                "text_padding_rows_nonzero": int(np.count_nonzero(
                    np.any(np.asarray(aligned["text"])[atb[1] == 0] != 0, axis=1)))}
        for modality, width in (("audio", 74), ("vision", 35)):
            target = np.asarray(aligned[modality])
            source = np.asarray(unaligned[modality])
            length_value = unaligned[f"{modality}_lengths"]
            if (target.shape != (50, width) or source.shape != (500, width)
                    or not isinstance(length_value, (int, np.integer))
                    or not 1 <= int(length_value) <= 500):
                raise ValueError(f"{sample_id}/{modality}: feature shape or length invalid")
            length = int(length_value)
            source_valid = source[:length]
            info[f"{modality}_unaligned_length"] = length
            info[f"{modality}_dtype_aligned"] = str(target.dtype)
            info[f"{modality}_dtype_unaligned"] = str(source.dtype)
            info[f"{modality}_source_padding_all_zero"] = bool(np.all(source[length:] == 0))
            info[f"{modality}_aligned_content_all_zero"] = bool(np.all(target[content] == 0))
            candidates_by_position = []
            for position in content:
                row = target[position]
                candidates = np.flatnonzero(np.all(source_valid == row, axis=1)).astype(int).tolist()
                candidates_by_position.append(candidates)
                row_matches.append({"sample_id": sample_id, "modality": modality,
                                    "official_seq_index": int(position),
                                    "aligned_feature_row_sha256": hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest(),
                                    "unaligned_valid_length": length,
                                    "exact_source_index_count": len(candidates),
                                    "exact_unaligned_indices_zero_based": ";".join(map(str, candidates)),
                                    "aligned_row_all_zero": bool(np.all(row == 0)),
                                    "raw_time_or_pts_verified": False})
            singles = [x[0] for x in candidates_by_position if len(x) == 1]
            all_unique = len(singles) == len(content)
            info[f"{modality}_rows_with_exact_source_match"] = sum(bool(x) for x in candidates_by_position)
            info[f"{modality}_rows_with_unique_source_match"] = len(singles)
            info[f"{modality}_all_unique_matches_nondecreasing"] = bool(
                all_unique and all(a <= b for a, b in zip(singles, singles[1:])))
            info[f"{modality}_unique_matches_form_valid_suffix"] = bool(
                all_unique and set(singles) == set(range(min(singles), length))) if singles else False
        inventory.append(info)
    write_csv(args.out / "pair_inventory_20.csv", inventory)
    write_csv(args.out / "aligned_to_unaligned_row_matches_20.csv", row_matches)
    vision_unmatched = [row for row in row_matches if row["modality"] == "vision"
                        and row["exact_source_index_count"] == 0]
    if vision_unmatched:
        write_csv(args.out / "vision_unmatched_rows.csv", vision_unmatched)
    by_modality = {}
    for modality in ("audio", "vision"):
        subset = [r for r in row_matches if r["modality"] == modality]
        by_modality[modality] = {
            "content_rows": len(subset),
            "rows_with_exact_unaligned_index": sum(r["exact_source_index_count"] > 0 for r in subset),
            "rows_with_unique_unaligned_index": sum(r["exact_source_index_count"] == 1 for r in subset),
            "rows_without_unaligned_index": sum(r["exact_source_index_count"] == 0 for r in subset),
            "samples_with_all_unique_matches_form_valid_suffix": sum(
                r[f"{modality}_unique_matches_form_valid_suffix"] for r in inventory),
            "raw_time_or_pts_verified": 0,
        }
    attachment_root = args.aligned.parent
    attachment_files = [path for path in attachment_root.rglob("*") if path.is_file()]
    source_code_or_config = [str(path.relative_to(attachment_root)) for path in attachment_files
                             if path.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".txt", ".md"}]
    summary = {
        "samples": 20,
        "problem_docx_sha256": sha256(args.problem_docx),
        "attachment4_file_extension_counts": dict(Counter(path.suffix.lower() or "<none>" for path in attachment_files)),
        "attachment4_code_or_config_files": source_code_or_config,
        "aligned_unaligned_video_bytes_equal": sum(r["video_bytes_equal"] for r in inventory),
        "raw_text_equal": sum(r["raw_text_equal"] for r in inventory),
        "text_feature_matrix_equal": sum(r["text_50x768_equal"] for r in inventory),
        "text_bert_equal": sum(r["text_bert_3x50_equal"] for r in inventory),
        "samples_with_nonzero_text_padding_rows": sum(r["text_padding_rows_nonzero"] > 0 for r in inventory),
        "vision_unmatched_rows_by_sample": dict(Counter(row["sample_id"] for row in vision_unmatched)),
        "by_modality": by_modality,
        "scope": "exact aligned-to-unaligned feature-row matching; no raw media time or frame provenance inferred",
        "model_loaded": False, "training_run": False, "predictions_computed": False,
    }
    (args.out / "pairwise_source_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
