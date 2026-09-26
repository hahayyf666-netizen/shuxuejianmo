from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct

from common import CANDIDATE, CLASS_NAMES, RESULTS, STAGE, write_csv, write_json

CORE = STAGE
sys.path.insert(0, str(CORE))
from q3v1.data import EXPECTED_SHA256, TrainScaler, load_split, sha256_file  # noqa: E402
from q3v1.model import Q3Model  # noqa: E402
from q3v1.train_eval import collate, metrics  # noqa: E402

EXPECTED = {
    "accuracy": 0.6442307692307693,
    "macro_f1": 0.6070737873693538,
    "per_class_f1": {"Negative": 0.6502463054187192, "Neutral": 0.4444444444444444, "Positive": 0.726530612244898},
    "mae": 0.5964004822775139,
    "pearson": 0.6290740140845968,
    "selection_J": 0.24616314650511592,
}
TOLERANCE = 1e-6
EXPECTED_CHECKPOINT_SHA256 = "723a9ddef831f35c25d95b325a51c194a8e7326460812ab5435c0b24fc8ad6ce"
EXPECTED_SCALER_SHA256 = "57d93f8d17fce56b928fadb1039bbe382a456386e39477980d44b43a2afd1424"
HIGH_CONFIDENCE_THRESHOLD = 0.80


def load_scaler(path: Path) -> TrainScaler:
    with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("source_split") != "train":
        raise ValueError("train scaler source is not train")
    scaler = TrainScaler()
    scaler.mean = {k: np.asarray(v, dtype=np.float64) for k, v in saved["mean"].items()}
    scaler.std = {k: np.asarray(v, dtype=np.float64) for k, v in saved["std"].items()}
    scaler.source_split = "train"
    return scaler


def distribution(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p10": None, "p25": None, "p75": None, "p90": None}
    return {"count": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
            "p10": float(np.quantile(arr, .10)), "p25": float(np.quantile(arr, .25)), "p75": float(np.quantile(arr, .75)), "p90": float(np.quantile(arr, .90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-pkl", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scaler", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is not available")

    # load_split enforces the aligned_50 hash and exposes only the requested valid split.
    valid = load_split(args.aligned_pkl, "valid", verify_hash=True)
    if len(valid) != 728:
        raise ValueError(f"expected 728 validation rows, received {len(valid)}")
    checkpoint_sha = sha256_file(args.checkpoint)
    scaler_sha = sha256_file(args.scaler)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256 or scaler_sha != EXPECTED_SCALER_SHA256:
        raise ValueError("STOP: checkpoint/scaler file identity differs from the frozen delivery artifacts")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "B0" or checkpoint.get("seed") != 2029 or checkpoint.get("epoch") != 2:
        raise ValueError("checkpoint identity is not frozen B0_seed2029 epoch 2")
    model = Q3Model("B0")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    scaler = load_scaler(args.scaler)

    records = []
    y_cls: list[int] = []
    p_cls: list[int] = []
    y_reg: list[float] = []
    p_reg: list[float] = []
    with torch.inference_mode():
        for begin in range(0, len(valid), 32):
            batch = valid[begin:begin + 32]
            text, audio, vision, mask, cls, reg = collate(batch, scaler, device)
            logits, intensity = model(text, audio, vision, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            predicted = logits.argmax(dim=-1).cpu().numpy()
            pred_intensity = intensity.cpu().numpy()
            true_class = cls.cpu().numpy()
            true_intensity = reg.cpu().numpy()
            for sample, pr, pc, pi, tc, ti in zip(batch, probs, predicted, pred_intensity, true_class, true_intensity):
                ranked = np.sort(pr)[::-1]
                records.append({
                    "sample_id": sample.sample_id, "true_class": CLASS_NAMES[int(tc)], "predicted_class": CLASS_NAMES[int(pc)],
                    "p_negative": float(pr[0]), "p_neutral": float(pr[1]), "p_positive": float(pr[2]),
                    "classification_margin": float(ranked[0] - ranked[1]), "top1_probability": float(ranked[0]),
                    "true_intensity": float(ti), "predicted_intensity": float(pi), "regression_error": float(pi - ti),
                    "absolute_regression_error": float(abs(pi - ti)), "classification_correct": int(tc == pc),
                })
                y_cls.append(int(tc)); p_cls.append(int(pc)); y_reg.append(float(ti)); p_reg.append(float(pi))

    met = metrics(np.asarray(y_cls), np.asarray(p_cls), np.asarray(y_reg), np.asarray(p_reg))
    errors = np.asarray(p_reg, dtype=np.float64) - np.asarray(y_reg, dtype=np.float64)
    met["rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    per_true = {}
    for index, label in enumerate(CLASS_NAMES):
        subset = [r for r in records if r["true_class"] == label]
        per_true[label] = {
            "count": len(subset),
            "classification_error_count": sum(1 - r["classification_correct"] for r in subset),
            "classification_error_rate": float(np.mean([1 - r["classification_correct"] for r in subset])) if subset else None,
            "regression_mae": float(np.mean([r["absolute_regression_error"] for r in subset])) if subset else None,
        }
    correct_margins = [r["classification_margin"] for r in records if r["classification_correct"]]
    incorrect_margins = [r["classification_margin"] for r in records if not r["classification_correct"]]
    direct = [r for r in records if (r["predicted_class"] == "Negative" and r["predicted_intensity"] > 0) or (r["predicted_class"] == "Positive" and r["predicted_intensity"] < 0)]
    conflicts_low_intensity = [r for r in direct if abs(r["predicted_intensity"]) < 0.2]
    conflicts_low_margin = [r for r in direct if r["classification_margin"] < 0.2]
    neutrals = [r for r in records if r["predicted_class"] == "Neutral"]
    high_conf_errors = [r for r in records if not r["classification_correct"] and r["top1_probability"] >= HIGH_CONFIDENCE_THRESHOLD]
    directions: dict[str, int] = {}
    for r in records:
        if not r["classification_correct"]:
            key = f"{r['true_class']}->{r['predicted_class']}"
            directions[key] = directions.get(key, 0) + 1
    by_abs_error = sorted(records, key=lambda r: (-r["absolute_regression_error"], r["sample_id"]))[:15]
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for truth, pred in zip(y_cls, p_cls):
        confusion[truth][pred] += 1

    checkpoint_valid = checkpoint.get("validation", {})
    metric_compare = {}
    keys = ["accuracy", "macro_f1", "mae", "pearson", "selection_J", "direct_opposite_polarity_rate", "neutral_predicted_abs_regression_mean"]
    for key in keys:
        observed, frozen, reference = met.get(key), checkpoint_valid.get(key), EXPECTED.get(key)
        checkpoint_pass = observed is not None and frozen is not None and abs(float(observed) - float(frozen)) <= TOLERANCE
        reference_required = key in EXPECTED
        reference_pass = (not reference_required) or (observed is not None and reference is not None and abs(float(observed) - float(reference)) <= TOLERANCE)
        metric_compare[key] = {"observed": observed, "checkpoint_validation": frozen, "expected_frozen_reference": reference,
                               "reference_comparison_required": reference_required,
                               "abs_diff_checkpoint": None if observed is None or frozen is None else abs(float(observed) - float(frozen)),
                               "abs_diff_reference": None if observed is None or reference is None else abs(float(observed) - float(reference)),
                               "pass_checkpoint": checkpoint_pass, "pass_reference": reference_pass, "pass": checkpoint_pass and reference_pass}
    for label in CLASS_NAMES:
        observed, frozen = met["per_class_f1"][label], checkpoint_valid.get("per_class_f1", {}).get(label)
        reference = EXPECTED["per_class_f1"][label]
        checkpoint_pass = abs(float(observed) - float(frozen)) <= TOLERANCE
        reference_pass = abs(float(observed) - float(reference)) <= TOLERANCE
        metric_compare[f"f1_{label}"] = {"observed": observed, "checkpoint_validation": frozen, "expected_frozen_reference": reference,
                                           "abs_diff_checkpoint": abs(float(observed) - float(frozen)), "abs_diff_reference": abs(float(observed) - float(reference)),
                                           "pass_checkpoint": checkpoint_pass, "pass_reference": reference_pass, "pass": checkpoint_pass and reference_pass}
    metric_gate = all(v["pass"] for v in metric_compare.values())

    fields = list(records[0])
    write_csv(out / "valid_predictions.csv", records, fields)
    valid_predictions_sha256 = sha256_file(out / "valid_predictions.csv")
    matrix_rows = [{"true_class": CLASS_NAMES[i], **{f"pred_{CLASS_NAMES[j]}": confusion[i][j] for j in range(3)}} for i in range(3)]
    write_csv(out / "valid_confusion_matrix.csv", matrix_rows, list(matrix_rows[0]))
    write_json(out / "valid_metrics.json", {
        "status": "VALID_FROZEN_INFERENCE_COMPLETE" if metric_gate else "STOP_VALID_METRIC_MISMATCH",
        "sample_count": len(records), "metrics": met, "confusion_matrix": {"class_order": list(CLASS_NAMES), "rows_true_cols_pred": confusion},
        "expected_frozen_validation": checkpoint_valid, "metric_comparison_tolerance": TOLERANCE, "metric_comparison": metric_compare,
        "checkpoint_identity": {"variant": checkpoint["variant"], "seed": checkpoint["seed"], "epoch": checkpoint.get("epoch")},
        "input_identity": {"aligned_pkl_sha256": EXPECTED_SHA256, "checkpoint_sha256": checkpoint_sha, "scaler_sha256": scaler_sha},
        "valid_predictions_sha256": valid_predictions_sha256,
        "metric_gate_requires_both_checkpoint_and_fixed_reference": True,
        "training_run": False, "optimizer_steps": 0, "split": "valid",
    })
    error = {
        "status": "PASS" if metric_gate else "STOP_VALID_METRIC_MISMATCH",
        "sample_count": len(records), "per_true_class": per_true,
        "classification_margin": {"correct": distribution(correct_margins), "incorrect": distribution(incorrect_margins)},
        "high_confidence_error_definition": f"incorrect class prediction with top-1 probability >= {HIGH_CONFIDENCE_THRESHOLD:.2f}",
        "high_confidence_error_count": len(high_conf_errors), "high_confidence_error_sample_ids": [r["sample_id"] for r in high_conf_errors],
        "direct_polarity_conflicts": {"count": len(direct), "low_intensity_abs_lt_0_2": len(conflicts_low_intensity), "low_margin_lt_0_2": len(conflicts_low_margin), "sample_ids": [r["sample_id"] for r in direct]},
        "predicted_neutral_count": len(neutrals), "predicted_neutral_abs_intensity_ge_0_2_count": sum(abs(r["predicted_intensity"]) >= 0.2 for r in neutrals),
        "main_confusion_directions": sorted(directions.items(), key=lambda item: (-item[1], item[0])),
        "largest_absolute_regression_errors_top15": by_abs_error,
        "threshold_tuning": False, "checkpoint_changed": False,
    }
    write_json(out / "valid_error_analysis.json", error)
    print(json.dumps({"status": error["status"], "sample_count": len(records), "metrics": met, "direct_conflicts": error["direct_polarity_conflicts"], "predicted_neutral": len(neutrals)}, ensure_ascii=False))
    if not metric_gate:
        raise SystemExit("STOP: observed Valid728 metrics differ from frozen checkpoint validation by more than 1e-6")


if __name__ == "__main__":
    main()
