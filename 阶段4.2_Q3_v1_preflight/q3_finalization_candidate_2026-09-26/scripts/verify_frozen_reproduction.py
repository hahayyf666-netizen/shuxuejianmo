from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import numpy as np

from common import CANDIDATE, RESULTS, STAGE, read_csv, read_json, write_json
sys.path.insert(0, str(STAGE))
from q3v1.data import EXPECTED_SHA256, sha256_file  # noqa: E402

TOL = 1e-6


def run(command: list[str], log: Path) -> int:
    proc = subprocess.run(command, cwd=STAGE.parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def compare_float(a, b) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= TOL


def compare_test(new_dir: Path, old_csv: Path, old_summary: Path) -> dict:
    new = {r["sample_id"]: r for r in read_csv(new_dir / "test_predictions.csv")}
    old = {r["sample_id"]: r for r in read_csv(old_csv)}
    problems = []
    if set(new) != set(old): problems.append({"type": "sample_id_set_mismatch", "new": len(new), "old": len(old)})
    checked = 0
    for sid in sorted(set(new) & set(old)):
        for key in ["predicted_class"]:
            if new[sid][key] != old[sid][key]: problems.append({"sample_id": sid, "field": key, "new": new[sid][key], "old": old[sid][key]})
        for key in ["p_negative", "p_neutral", "p_positive", "predicted_intensity"]:
            if not compare_float(new[sid][key], old[sid][key]): problems.append({"sample_id": sid, "field": key, "new": new[sid][key], "old": old[sid][key]})
            checked += 1
    nsummary, osummary = read_json(new_dir / "test_evaluation_summary.json"), read_json(old_summary)
    metric_problems = {}
    for key in ["accuracy", "macro_f1", "mae", "rmse", "pearson"]:
        a, b = nsummary["metrics"].get(key), osummary["metrics"].get(key)
        ok = a is not None and b is not None and compare_float(a, b)
        metric_problems[key] = {"new": a, "historical": b, "pass": ok}
    return {"pass": not problems and all(x["pass"] for x in metric_problems.values()) and len(new) == 727,
            "sample_count": len(new), "numeric_values_compared": checked, "tolerance_abs": TOL,
            "row_mismatch_count": len(problems), "row_mismatches_first20": problems[:20], "metric_comparison": metric_problems}


def compare_attachment4(new_dir: Path, old_results: Path) -> dict:
    errors = []
    compared = 0
    for i in range(1, 21):
        sid = f"{i:02d}"
        new = read_json(new_dir / "samples" / f"sample_{sid}.json")
        old = read_json(old_results / "samples" / f"sample_{sid}.json")
        npred, opred = new["prediction"], old["prediction"]
        if npred["class_index"] != opred["class_index"]: errors.append({"sample_id": sid, "field": "class_index"})
        for klass in ("Negative", "Neutral", "Positive"):
            if not compare_float(npred["probabilities"][klass], opred["probabilities"][klass]): errors.append({"sample_id": sid, "field": f"p_{klass}"})
        if not compare_float(npred["intensity"], opred["intensity"]): errors.append({"sample_id": sid, "field": "intensity"})
        for target in ("classification", "regression"):
            nphi, ophi = new["targets"][target]["shapley"]["phi"], old["targets"][target]["shapley"]["phi"]
            for modality in ("text", "audio", "vision"):
                if not compare_float(nphi[modality], ophi[modality]): errors.append({"sample_id": sid, "target": target, "field": f"shapley_{modality}"})
            for modality in ("text", "audio", "vision"):
                ni = new["targets"][target]["modalities"][modality]["conditional_ig"]
                oi = old["targets"][target]["modalities"][modality]["conditional_ig"]
                if ni["numerical_status"] != "pass" or oi["numerical_status"] != "pass": errors.append({"sample_id": sid, "target": target, "modality": modality, "field": "ig_status"})
                ntop = new["targets"][target]["modalities"][modality]["top_positions"]
                otop = old["targets"][target]["modalities"][modality]["top_positions"]
                nmap = {int(x["official_seq_index"]): float(x["signed_importance"]) for x in ntop}
                omap = {int(x["official_seq_index"]): float(x["signed_importance"]) for x in otop}
                if set(nmap) != set(omap): errors.append({"sample_id": sid, "target": target, "modality": modality, "field": "top_position_set", "new": sorted(nmap), "historical": sorted(omap)})
                else:
                    for idx in nmap:
                        if not compare_float(nmap[idx], omap[idx]): errors.append({"sample_id": sid, "target": target, "modality": modality, "field": f"top_signed_importance_{idx}"})
                compared += 1
    return {"pass": not errors, "sample_count": 20, "top_position_sets_compared": compared, "tolerance_abs": TOL,
            "mismatch_count": len(errors), "mismatches_first30": errors[:30]}


def recompute_valid_csv(path: Path) -> dict:
    rows = read_csv(path)
    if len(rows) != 728 or len({r["sample_id"] for r in rows}) != 728:
        raise ValueError("valid_predictions.csv must have 728 unique sample IDs")
    labels = ["Negative", "Neutral", "Positive"]
    yi = np.asarray([labels.index(r["true_class"]) for r in rows], dtype=np.int64)
    pi = np.asarray([labels.index(r["predicted_class"]) for r in rows], dtype=np.int64)
    yt = np.asarray([float(r["true_intensity"]) for r in rows], dtype=np.float64)
    yp = np.asarray([float(r["predicted_intensity"]) for r in rows], dtype=np.float64)
    confusion = [[int(np.sum((yi == i) & (pi == j))) for j in range(3)] for i in range(3)]
    f1 = {}
    for i, label in enumerate(labels):
        tp = confusion[i][i]; fp = sum(confusion[r][i] for r in range(3) if r != i); fn = sum(confusion[i][c] for c in range(3) if c != i)
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    macro_f1 = float(np.mean(list(f1.values())))
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean(np.square(yp - yt))))
    pearson = float(np.corrcoef(yt, yp)[0, 1])
    return {"sample_count": len(rows), "sample_id_unique_count": len({r["sample_id"] for r in rows}),
            "metrics": {"accuracy": float(np.mean(yi == pi)), "macro_f1": macro_f1, "per_class_f1": f1, "mae": mae,
                        "rmse": rmse, "pearson": pearson, "selection_J": 0.5 * (1 - macro_f1) + 0.5 * mae / 6.0},
            "confusion_matrix": confusion}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-pkl", type=Path, required=True)
    ap.add_argument("--attachment-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scaler", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--text-trace", type=Path, required=True)
    ap.add_argument("--t3-records", type=Path, required=True)
    ap.add_argument("--historical-test-predictions", type=Path, required=True)
    ap.add_argument("--historical-test-summary", type=Path, required=True)
    ap.add_argument("--historical-attachment-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=CANDIDATE / "reproduction_tmp")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"reproduction output must be empty: {out}")
    test_out, attach_out = out / "test", out / "attachment4"
    test_out.mkdir(parents=True, exist_ok=True); attach_out.mkdir(parents=True, exist_ok=True)
    test_script = STAGE / "run_final_test_evaluation.py"
    attach_script = STAGE / "formal_attachment4_2026-09-26/run_attachment4_formal.py"
    code1 = run([sys.executable, str(test_script), "--aligned-pkl", str(args.aligned_pkl.resolve()), "--checkpoint", str(args.checkpoint.resolve()), "--scaler", str(args.scaler.resolve()), "--out", str(test_out), "--device", "cpu"], out / "test.stdout.log")
    code2 = run([sys.executable, str(attach_script), "--attachment-dir", str(args.attachment_dir.resolve()), "--checkpoint", str(args.checkpoint.resolve()), "--scaler", str(args.scaler.resolve()), "--contract", str(args.contract.resolve()), "--text-trace", str(args.text_trace.resolve()), "--t3-records", str(args.t3_records.resolve()), "--out", str(attach_out), "--device", "cpu"], out / "attachment4.stdout.log")
    stage_e = read_json(RESULTS / "valid_metrics.json")
    test_result = compare_test(test_out, args.historical_test_predictions, args.historical_test_summary) if code1 == 0 else {"pass": False, "runner_exit_code": code1}
    attach_result = compare_attachment4(attach_out, args.historical_attachment_results) if code2 == 0 else {"pass": False, "runner_exit_code": code2}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    valid_metric = stage_e.get("metric_comparison", {})
    input_identity = stage_e.get("input_identity", {})
    valid_csv = RESULTS / "valid_predictions.csv"
    valid_csv_sha = sha256_file(valid_csv) if valid_csv.is_file() else None
    valid_recomputed = recompute_valid_csv(valid_csv) if valid_csv.is_file() else {"sample_count": 0, "sample_id_unique_count": 0, "metrics": {}}
    expected_metric_keys = {"accuracy", "macro_f1", "mae", "pearson", "selection_J", "direct_opposite_polarity_rate", "neutral_predicted_abs_regression_mean", "f1_Negative", "f1_Neutral", "f1_Positive"}
    actual_input_identity = {"aligned_pkl_sha256": EXPECTED_SHA256, "checkpoint_sha256": sha256_file(args.checkpoint), "scaler_sha256": sha256_file(args.scaler)}
    valid_identity_pass = input_identity == actual_input_identity and actual_input_identity["aligned_pkl_sha256"] == EXPECTED_SHA256
    valid_artifact_pass = (stage_e.get("sample_count") == 728 and valid_recomputed.get("sample_count") == 728 and
                           valid_recomputed.get("sample_id_unique_count") == 728 and valid_csv_sha == stage_e.get("valid_predictions_sha256"))
    valid_csv_metric_keys = ["accuracy", "macro_f1", "mae", "pearson", "selection_J"]
    valid_csv_metrics_pass = all(k in stage_e.get("metrics", {}) and k in valid_recomputed.get("metrics", {}) and compare_float(valid_recomputed["metrics"][k], stage_e["metrics"][k]) for k in valid_csv_metric_keys)
    valid_pass = (stage_e.get("status") == "VALID_FROZEN_INFERENCE_COMPLETE" and valid_identity_pass and valid_artifact_pass and valid_csv_metrics_pass and
                  expected_metric_keys.issubset(valid_metric) and all(valid_metric[k].get("pass") and valid_metric[k].get("pass_reference") and valid_metric[k].get("pass_checkpoint") for k in expected_metric_keys))
    overall = "PASS" if valid_pass and test_result.get("pass") and attach_result.get("pass") else "FAIL"
    result = {
        "status": "FROZEN_REPRODUCTION_COMPLETE" if overall == "PASS" else "FROZEN_REPRODUCTION_FAILED",
        "reproduction_gate": overall,
        "interpretation": "current frozen implementation reproduces frozen outputs" if overall == "PASS" else "one or more frozen reproduction comparisons failed",
        "historical_exact_invocation_provenance_proven": False,
        "valid_reproduced": valid_pass, "test_reproduced": bool(test_result.get("pass")), "attachment4_reproduced": bool(attach_result.get("pass")),
        "valid": {"metric_comparison": valid_metric, "input_identity": input_identity, "current_input_identity": actual_input_identity,
                  "identity_pass": valid_identity_pass, "valid_artifact_pass": valid_artifact_pass,
                  "valid_csv_sha256": valid_csv_sha, "recorded_valid_csv_sha256": stage_e.get("valid_predictions_sha256"),
                  "recomputed_from_valid_csv": valid_recomputed, "valid_csv_metrics_match_record": valid_csv_metrics_pass, "tolerance_abs": TOL}, "test": test_result, "attachment4": attach_result,
        "training_run": False, "optimizer_steps": 0, "valid120_xai_rerun": False, "t4_rerun": False,
        "checkpoint_identity": {"variant": checkpoint.get("variant"), "seed": checkpoint.get("seed"), "epoch": checkpoint.get("epoch")},
        "test_used_for_tuning": False, "historical_results_overwritten": False,
    }
    write_json(RESULTS / "reproduction_check.json", result)
    print(json.dumps({"reproduction_gate": overall, "valid": valid_pass, "test": result["test_reproduced"], "attachment4": result["attachment4_reproduced"]}, ensure_ascii=False))
    if overall != "PASS":
        raise SystemExit("STOP: frozen reproduction gate failed; do not package or commit as ready")


if __name__ == "__main__":
    main()
