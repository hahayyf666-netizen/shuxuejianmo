"""Portable frozen Q3 Attachment4 inference and feature-space explanation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from q3v1.data import validate_content_indices  # noqa: E402
from q3v1.model import Q3Model  # noqa: E402
from q3v1.explain import conditional_ig, exact_shapley, local_attribution_status, summarize_contributions  # noqa: E402

MODALITIES = ("text", "audio", "vision")
CLASS_NAMES = ("Negative", "Neutral", "Positive")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scaler(path: Path) -> dict:
    with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("source_split") != "train":
        raise ValueError("scaler source is not train")
    return saved


def load_sample(path: Path) -> dict:
    with path.open("rb") as f:
        row = pickle.load(f)
    required = {"raw_text", "id", "text", "audio", "vision", "text_bert"}
    if not required.issubset(row):
        raise ValueError(f"{path.name}: missing {sorted(required - set(row))}")
    text_bert = np.asarray(row["text_bert"])
    content = validate_content_indices(text_bert)
    values = {k: np.asarray(row[k], dtype=np.float32) for k in MODALITIES}
    expected = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35)}
    if any(values[k].shape != expected[k] for k in MODALITIES):
        raise ValueError(f"{path.name}: feature shape mismatch")
    if any(not v.flags["C_CONTIGUOUS"] and False for v in values.values()):
        raise ValueError(f"{path.name}: unexpected feature storage")
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError(f"{path.name}: nonfinite feature")
    return {
        "sample_id": str(path.stem),
        "official_id": str(row["id"]),
        "raw_text": str(row["raw_text"]),
        "content_indices": content,
        **values,
    }


def prepare(sample: dict, scaler: dict, device: torch.device):
    mask = np.zeros(50, dtype=bool)
    mask[sample["content_indices"]] = True
    transformed = []
    for modality in MODALITIES:
        mean = np.asarray(scaler["mean"][modality], dtype=np.float64)
        std = np.asarray(scaler["std"][modality], dtype=np.float64)
        arr = ((sample[modality].astype(np.float64) - mean) / std).astype(np.float32)
        arr[~mask] = 0
        transformed.append(torch.from_numpy(arr[None]).to(device))
    mask_t = torch.from_numpy(mask[None]).to(device)
    refs = tuple(torch.zeros_like(x) for x in transformed)
    return tuple(transformed), mask_t, refs, mask


def read_text_trace(path: Path) -> dict:
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("strict_verified_within_1e-4", "").lower() != "true":
                continue
            out[(str(row["sample_id"]), int(row["official_seq_index"]))] = {
                "token_id": int(row["token_id"]),
                "char_start": int(row["char_start"]),
                "char_end": int(row["char_end"]),
                "raw_text_substring": row["raw_text_substring"],
            }
    return out


def top_positions(scores: np.ndarray, content_indices: np.ndarray, mapping: str, trace: dict, sid: str, top_k: int):
    ranked = sorted((int(i) for i in content_indices), key=lambda i: (-abs(float(scores[i])), i))[:top_k]
    result = []
    for idx in ranked:
        rec = {
            "official_seq_index": idx,
            "signed_importance": float(scores[idx]),
            "absolute_importance": float(abs(scores[idx])),
            "mapping_status": mapping,
        }
        if mapping == "verified_text" and (sid, idx) in trace:
            rec.update(trace[(sid, idx)])
        result.append(rec)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attachment-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scaler", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--text-trace", type=Path, required=True)
    ap.add_argument("--t3-records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    started = time.monotonic()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["model"] != "B0_seed2029" or contract["prediction_contract"]["labels_read"]:
        raise ValueError("Attachment4 contract drift")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")

    scaler = load_scaler(args.scaler)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "B0" or checkpoint.get("seed") != 2029:
        raise ValueError("checkpoint identity mismatch")

    model = Q3Model("B0")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)

    trace = read_text_trace(args.text_trace)
    t3 = {str(r["sample_id"]): r for r in json.loads(args.t3_records.read_text(encoding="utf-8"))}

    sample_paths = sorted(args.attachment_dir.glob("[0-9][0-9].pkl"))
    if [p.stem for p in sample_paths] != [f"{i:02d}" for i in range(1, 21)]:
        raise ValueError("Attachment4 must contain exactly 01-20")

    records = []
    local_rows = []
    numeric_checks = []
    (args.out / "samples").mkdir(exist_ok=True)

    for ordinal, path in enumerate(sample_paths, 1):
        sample = load_sample(path)
        xs, mask, refs, mask_np = prepare(sample, scaler, device)

        with torch.no_grad():
            logits, intensity = model(*xs, mask)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            target_class = int(np.argmax(probs))
            pred_intensity = float(intensity.item())

        sid = sample["sample_id"]
        record = {
            "sample_id": sid,
            "official_id": sample["official_id"],
            "raw_text": sample["raw_text"],
            "valid_content_length": int(len(sample["content_indices"])),
            "official_seq_indices": [int(i) for i in sample["content_indices"]],
            "prediction": {
                "class_index": target_class,
                "class_name": CLASS_NAMES[target_class],
                "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
                "intensity": pred_intensity,
            },
            "mapping_status": {"text": "verified_text", "audio": "index_only", "vision": "index_only"},
            "media_origin_audit": t3.get(sid, {}).get("local_time_gate", "NOT_TESTED"),
            "known_input_anomaly": contract["card_selection"]["known_input_anomalies"].get(sid),
            "targets": {},
        }

        for target in ("classification", "regression"):
            shap = exact_shapley(model, xs, refs, mask, target=target, target_class=target_class)
            shap_pass = abs(float(shap["additivity_residual"])) <= 1e-6
            numeric_checks.append({
                "sample_id": sid,
                "target": target,
                "method": "shapley",
                "pass": shap_pass,
                "residual": shap["additivity_residual"],
            })

            target_record = {
                "shapley": shap,
                "semantics": summarize_contributions(shap["phi"], target=target, epsilon=1e-6),
                "modalities": {},
            }

            for modality in MODALITIES:
                ig = conditional_ig(
                    model, xs, refs, mask,
                    modality=modality,
                    target=target,
                    target_class=target_class,
                    steps_schedule=(64, 128, 256),
                    atol=1e-3,
                    rtol=0.01,
                )
                numeric_checks.append({
                    "sample_id": sid,
                    "target": target,
                    "modality": modality,
                    "method": "conditional_ig",
                    "pass": ig["numerical_status"] == "pass",
                    "residual": ig["completeness_residual"],
                    "steps": ig["steps"],
                })

                mapping = "verified_text" if modality == "text" else "index_only"
                positions = top_positions(
                    ig["position_scores"],
                    sample["content_indices"],
                    mapping,
                    trace,
                    sid,
                    int(contract["prediction_contract"]["local_top_k"]),
                )

                for p in positions:
                    local_rows.append({
                        "sample_id": sid,
                        "target": target,
                        "modality": modality,
                        "official_seq_index": p["official_seq_index"],
                        "signed_importance": p["signed_importance"],
                        "absolute_importance": p["absolute_importance"],
                        "mapping_status": p["mapping_status"],
                        "token_id": p.get("token_id"),
                        "char_start": p.get("char_start"),
                        "char_end": p.get("char_end"),
                        "raw_text_substring": p.get("raw_text_substring"),
                    })

                target_record["modalities"][modality] = {
                    "conditional_ig": {k: v for k, v in ig.items() if k != "position_scores"},
                    "local_attribution_status": local_attribution_status(
                        shap["phi"][modality], ig, mask_np, epsilon=1e-6
                    ),
                    "top_positions": positions,
                }

            record["targets"][target] = target_record

        records.append(record)
        (args.out / "samples" / f"sample_{sid}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{ordinal}/20] {sid} class={CLASS_NAMES[target_class]} intensity={pred_intensity:.6f}", flush=True)

    (args.out / "attachment4_predictions_explanations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "sample_id", "target", "modality", "official_seq_index",
        "signed_importance", "absolute_importance", "mapping_status",
        "token_id", "char_start", "char_end", "raw_text_substring",
    ]
    with (args.out / "local_importance_top5.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(local_rows)

    summary = {
        "status": "ATTACHMENT4_INFERENCE_COMPLETE",
        "sample_count": len(records),
        "numeric_check_count": len(numeric_checks),
        "numeric_failures": [x for x in numeric_checks if not x["pass"]],
        "numerical_status": "PASS" if all(x["pass"] for x in numeric_checks) else "REVIEW_REQUIRED",
        "mapping_status": {"text": "verified_text", "audio": "index_only", "vision": "index_only"},
        "feature_space_attribution": True,
        "labels_read": False,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "scaler_sha256": sha256_file(args.scaler),
        "elapsed_sec": time.monotonic() - started,
    }
    (args.out / "numeric_checks.json").write_text(
        json.dumps(numeric_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
