"""Audit Attachment4 raw-text to text_bert offsets; never loads a predictor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import transformers
from transformers import BertTokenizerFast

TOKENIZER_REPOSITORY = "google-bert/bert-base-uncased"
TOKENIZER_REVISION = "b96743c503420c0858ad23fca994e670844c6c05"
EXPECTED_VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_text_bert(value: object, *, sample_id: str, vocab_size: int) -> np.ndarray:
    """Validate the original array before any conversion to integer token IDs."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: text_bert cannot be represented as an array") from exc
    if raw.shape != (3, 50):
        raise ValueError(f"{sample_id}: text_bert shape must be (3, 50), got {raw.shape}")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{sample_id}: text_bert must have a real numeric dtype, got {raw.dtype}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{sample_id}: text_bert contains nonfinite values")
    if raw.dtype.kind == "f" and not np.array_equal(raw, np.trunc(raw)):
        raise ValueError(f"{sample_id}: text_bert contains noninteger values")
    if np.any(raw[0] < 0) or np.any(raw[0] >= vocab_size):
        raise ValueError(f"{sample_id}: text_bert token ID outside tokenizer vocabulary")
    if not np.all((raw[1] == 0) | (raw[1] == 1)):
        raise ValueError(f"{sample_id}: text_bert attention mask must be binary")
    if not np.all(raw[2] == 0):
        raise ValueError(f"{sample_id}: text_bert segment IDs must be zero for one input text")
    active_length = int(np.count_nonzero(raw[1]))
    if (active_length < 3 or not np.all(raw[1, :active_length] == 1)
            or not np.all(raw[1, active_length:] == 0)):
        raise ValueError(f"{sample_id}: text_bert attention mask must contain an active content prefix")
    if raw[0, 0] != 101 or raw[0, active_length - 1] != 102:
        raise ValueError(f"{sample_id}: text_bert must start with CLS and end the active prefix with SEP")
    if np.any(raw[0, 1:active_length - 1] == 102):
        raise ValueError(f"{sample_id}: text_bert has an unexpected SEP inside content positions")
    if not np.all(raw[0, active_length:] == 0):
        raise ValueError(f"{sample_id}: text_bert padding token IDs must be zero")
    return raw.astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--attachment4-aligned", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vocab_sha = sha256(args.vocab)
    if vocab_sha != EXPECTED_VOCAB_SHA256:
        raise ValueError(f"vocab SHA mismatch: {vocab_sha}")
    tokenizer = BertTokenizerFast(vocab_file=str(args.vocab), do_lower_case=True, model_max_length=50)
    summary_rows, offset_rows = [], []
    for number in range(1, 21):
        sample_id = f"{number:02d}"
        pkl = args.attachment4_aligned / f"{sample_id}.pkl"
        with pkl.open("rb") as stream:
            item = pickle.load(stream)
        if str(item.get("id")) != sample_id:
            raise ValueError(f"sample ID mismatch: {sample_id}")
        official = validate_text_bert(item["text_bert"], sample_id=sample_id,
                                      vocab_size=tokenizer.vocab_size)
        raw_text = str(item["raw_text"])
        encoded = tokenizer(raw_text, truncation=True, max_length=50, padding="max_length",
                            return_offsets_mapping=True)
        exact_ids = np.array_equal(official[0], encoded["input_ids"])
        exact_mask = np.array_equal(official[1], encoded["attention_mask"])
        exact_segments = np.array_equal(official[2], encoded["token_type_ids"])
        active = np.flatnonzero(official[1] == 1)
        if not exact_ids or not exact_mask or not exact_segments:
            mapping_status = "index_only"
        else:
            mapping_status = "token_span_candidate_only"
        for idx in active[1:-1]:
            start, end = encoded["offset_mapping"][int(idx)]
            offset_rows.append({
                "sample_id": sample_id,
                "official_seq_index": int(idx),
                "token_id": int(official[0, idx]),
                "token": tokenizer.convert_ids_to_tokens(int(official[0, idx])),
                "char_start": int(start), "char_end": int(end),
                "raw_text_substring": raw_text[start:end],
                "ids_exact": exact_ids, "attention_exact": exact_mask,
                "segments_exact": exact_segments,
                "mapping_status": mapping_status,
                "scope_limit": "text_bert token row only; continuous text feature row provenance not established",
            })
        keys = sorted(item.keys())
        temporal_keys = [key for key in keys if any(term in key.lower() for term in
                         ("time", "start", "end", "pts", "frame", "timestamp", "align"))]
        video = args.attachment4_aligned / "videos" / f"{sample_id}.mp4"
        vision = np.asarray(item["vision"])
        summary_rows.append({
            "sample_id": sample_id,
            "pkl_sha256": sha256(pkl),
            "video_sha256": sha256(video) if video.is_file() else "",
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "pkl_keys": ";".join(keys),
            "per_row_temporal_fields": ";".join(temporal_keys),
            "token_count_excluding_specials": int(max(len(active) - 2, 0)),
            "ids_exact": exact_ids, "attention_exact": exact_mask,
            "segments_exact": exact_segments,
            "vision_all_zero": bool(np.all(vision == 0)),
            "video_present": video.is_file(),
            "token_span_status": mapping_status,
        })
    args.out.mkdir(parents=True, exist_ok=True)
    def write_csv(name: str, rows: list[dict]) -> None:
        with (args.out / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_csv("attachment4_mapping_inventory_20.csv", summary_rows)
    write_csv("text_tokenizer_replay_20.csv", offset_rows)
    summary = {
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "vocab_sha256": vocab_sha,
        "transformers_version": transformers.__version__,
        "settings": {"do_lower_case": True, "max_length": 50, "truncation": True,
                     "padding": "max_length", "offset_mapping": True},
        "samples": len(summary_rows),
        "ids_attention_segments_exact": sum(r["ids_exact"] and r["attention_exact"] and r["segments_exact"] for r in summary_rows),
        "text_bert_content_positions_with_offsets": len(offset_rows),
        "video_pairs_present": sum(r["video_present"] for r in summary_rows),
        "vision_all_zero_samples": [r["sample_id"] for r in summary_rows if r["vision_all_zero"]],
        "temporal_metadata_key_samples": sum(bool(r["per_row_temporal_fields"]) for r in summary_rows),
        "model_loaded": False,
        "predictions_computed": False,
        "interpretation": "Exact tokenizer replay supports candidate raw_text character spans for text_bert token positions. It does not establish continuous text embedding row provenance. PKLs have no obvious per-row temporal/PTS/frame fields; raw MP4 presence alone does not establish aligned audio/vision row support.",
    }
    (args.out / "tokenizer_replay_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
