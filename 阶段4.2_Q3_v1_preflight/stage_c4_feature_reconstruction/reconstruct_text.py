"""Reconstruct Attachment 4 text rows from raw text with a pinned BERT candidate.

The candidate is tested against the official matrix; its identity is not assumed.
No Q3 prediction model is loaded or trained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_c1_evidence_mapping"))
from probe_tokenizer_mapping import validate_text_bert  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(official: np.ndarray, candidate: np.ndarray, positions: np.ndarray) -> dict:
    truth = official[positions].astype(np.float64)
    test = candidate[positions].astype(np.float64)
    difference = np.abs(truth - test)
    dot = np.sum(truth * test, axis=1)
    norms = np.linalg.norm(truth, axis=1) * np.linalg.norm(test, axis=1)
    cosine = np.divide(dot, norms, out=np.full_like(dot, np.nan), where=norms > 0)
    per_row_max = np.max(difference, axis=1)
    return {
        "row_count": int(len(positions)),
        "mean_row_cosine": float(np.nanmean(cosine)),
        "median_row_cosine": float(np.nanmedian(cosine)),
        "mean_absolute_error": float(np.mean(difference)),
        "max_absolute_error": float(np.max(difference)),
        "rows_all_dims_within_1e-4": int(np.sum(per_row_max <= 1e-4)),
        "rows_all_dims_within_1e-3": int(np.sum(per_row_max <= 1e-3)),
        "rows_equal_after_4_decimal_rounding": int(np.sum(np.all(
            np.round(test, 4) == np.round(truth, 4), axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = BertTokenizerFast(vocab_file=str(args.vocab), do_lower_case=True, model_max_length=50)
    model = BertModel.from_pretrained(str(args.model_dir), add_pooling_layer=False,
                                     output_hidden_states=True, local_files_only=True)
    model.eval()
    records = []
    row_traces = []
    selected = None
    calibration_grid = []
    for number in range(1, 21):
        sid = f"{number:02d}"
        pkl = args.aligned / f"{sid}.pkl"
        with pkl.open("rb") as stream:
            item = pickle.load(stream)
        if str(item.get("id")) != sid:
            raise ValueError(f"{sid}: unexpected sample identity")
        raw_text = str(item["raw_text"])
        encoded = tokenizer(raw_text, truncation=True, max_length=50,
                            padding="max_length", return_offsets_mapping=True,
                            return_tensors="pt")
        offsets = encoded.pop("offset_mapping")[0].numpy()
        official_tokens = validate_text_bert(item["text_bert"], sample_id=sid,
                                             vocab_size=tokenizer.vocab_size)
        replay = np.stack([encoded["input_ids"][0].numpy(),
                           encoded["attention_mask"][0].numpy(),
                           encoded["token_type_ids"][0].numpy()])
        if not np.array_equal(replay, official_tokens):
            raise ValueError(f"{sid}: raw text tokenizer replay differs from official text_bert")
        official = np.asarray(item["text"], dtype=np.float32)
        if official.shape != (50, 768) or not np.isfinite(official).all():
            raise ValueError(f"{sid}: official text matrix malformed")
        active = np.flatnonzero(official_tokens[1] == 1)
        content = active[1:-1]
        candidates = []
        for mask_name in ("official_attention", "all_ones"):
            inputs = {name: value for name, value in encoded.items()}
            if mask_name == "all_ones":
                inputs["attention_mask"] = torch.ones_like(inputs["attention_mask"])
            with torch.inference_mode():
                hidden = model(**inputs).hidden_states
            for layer, value in enumerate(hidden):
                matrix = value[0].cpu().numpy()
                metrics = compare(official, matrix, content)
                candidates.append({"mask": mask_name, "layer": layer, **metrics})
        if selected is None:
            calibration_grid = sorted(candidates, key=lambda x: x["mean_absolute_error"])
            selected = {"mask": calibration_grid[0]["mask"], "layer": calibration_grid[0]["layer"]}
        chosen = next(row for row in candidates if row["mask"] == selected["mask"]
                      and row["layer"] == selected["layer"])
        selected_inputs = {name: value for name, value in encoded.items()}
        if selected["mask"] == "all_ones":
            selected_inputs["attention_mask"] = torch.ones_like(selected_inputs["attention_mask"])
        with torch.inference_mode():
            selected_matrix = model(**selected_inputs).hidden_states[selected["layer"]][0].cpu().numpy()
        all_position_metrics = compare(official, selected_matrix, np.arange(50))
        for position in content:
            index = int(position)
            official_row = official[index].astype(np.float64)
            reconstructed_row = selected_matrix[index].astype(np.float64)
            absolute = np.abs(official_row - reconstructed_row)
            start, end = (int(value) for value in offsets[index])
            row_traces.append({
                "sample_id": sid,
                "official_seq_index": index,
                "token_id": int(official_tokens[0, index]),
                "char_start": start,
                "char_end": end,
                "raw_text_substring": raw_text[start:end],
                "official_row_sha256": hashlib.sha256(
                    np.ascontiguousarray(official[index]).tobytes()).hexdigest(),
                "reconstructed_row_sha256": hashlib.sha256(
                    np.ascontiguousarray(selected_matrix[index]).tobytes()).hexdigest(),
                "row_cosine": float(np.dot(official_row, reconstructed_row) /
                                    (np.linalg.norm(official_row) *
                                     np.linalg.norm(reconstructed_row))),
                "row_mean_absolute_error": float(np.mean(absolute)),
                "row_max_absolute_error": float(np.max(absolute)),
                "strict_verified_within_1e-4": bool(np.max(absolute) <= 1e-4),
            })
        records.append({"sample_id": sid, "pkl_sha256": sha256(pkl),
                        "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                        "content_rows": int(len(content)),
                        "official_four_decimal_max_residual": float(np.max(
                            np.abs(official - np.round(official, 4)))),
                        **chosen, "all_50_position_metrics": all_position_metrics})
        print(sid, chosen["mean_row_cosine"], chosen["mean_absolute_error"],
              chosen["rows_all_dims_within_1e-4"], "/", len(content), flush=True)
    total_rows = sum(row["content_rows"] for row in records)
    trace_path = args.out / "text_row_trace_564.csv"
    with trace_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_traces[0]))
        writer.writeheader()
        writer.writerows(row_traces)
    summary = {
        "candidate": "google-bert/bert-base-uncased BertModel hidden states",
        "candidate_revision": "8229d58a8e9c4f761cdb4a3f0434f856e1ae1d5d",
        "model_safetensors_sha256": sha256(args.model_dir / "model.safetensors"),
        "config_sha256": sha256(args.model_dir / "config.json"),
        "vocab_sha256": sha256(args.vocab),
        "candidate_selection": "minimum content-row MAE on sample 01 among 13 layers x 2 attention variants; samples 02-20 held out from selection",
        "selected_variant": selected,
        "calibration_sample_01_grid": calibration_grid,
        "sample_metrics": records,
        "total_content_rows": total_rows,
        "total_rows_all_dims_within_1e-4": sum(row["rows_all_dims_within_1e-4"] for row in records),
        "total_rows_equal_after_4_decimal_rounding": sum(
            row["rows_equal_after_4_decimal_rounding"] for row in records),
        "total_all_50_positions_within_1e-4": sum(
            row["all_50_position_metrics"]["rows_all_dims_within_1e-4"] for row in records),
        "text_row_trace_csv_sha256": sha256(trace_path),
        "training_run": False,
        "q3_predictor_loaded": False,
    }
    (args.out / "text_reconstruction.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
