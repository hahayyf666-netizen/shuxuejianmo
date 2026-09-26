"""Recompute frozen test metrics from the existing per-sample CSV; no inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CLASSES = ("Negative", "Neutral", "Positive")
EXPECTED_CSV_SHA256 = "454e62de2c3704823f65bb13bb92ae04bff9d8a205e28e0428ee4d7a0f11b3b5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pred_path = args.predictions
    summary_path = args.summary
    manifest_path = args.manifest
    with pred_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = manifest_path.parent
    manifest_ok = all((manifest_root / name).is_file() and sha256(manifest_root / name) == digest for name, digest in manifest.items())
    ids = [row["sample_id"] for row in rows]
    ids_unique = len(ids) == len(set(ids))
    cm = [[0] * 3 for _ in range(3)]
    ys, ps, errors, prob_failures, finite_failures = [], [], [], [], []
    for i, row in enumerate(rows):
        y_name, p_name = row["true_class"], row["predicted_class"]
        yi, pi = CLASSES.index(y_name), CLASSES.index(p_name)
        cm[yi][pi] += 1
        y, p = float(row["true_intensity"]), float(row["predicted_intensity"])
        ys.append(y)
        ps.append(p)
        errors.append(p - y)
        probs = [float(row[f"p_{name.lower()}"]) for name in ("Negative", "Neutral", "Positive")]
        if not all(math.isfinite(x) and -1e-12 <= x <= 1 + 1e-12 for x in probs) or abs(sum(probs) - 1.0) > 1e-6:
            prob_failures.append(i)
        if not all(math.isfinite(x) for x in (y, p)):
            finite_failures.append(i)
    n = len(rows)
    acc = sum(cm[i][i] for i in range(3)) / n if n else float("nan")
    f1 = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3) if j != i)
        fn = sum(cm[i][j] for j in range(3) if j != i)
        f1[name] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    macro = sum(f1.values()) / 3
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    ym, pm = sum(ys) / n, sum(ps) / n
    covariance = sum((a - ym) * (b - pm) for a, b in zip(ys, ps))
    vy = sum((a - ym) ** 2 for a in ys)
    vp = sum((b - pm) ** 2 for b in ps)
    pearson = covariance / math.sqrt(vy * vp) if vy > 0 and vp > 0 else None
    metrics = {"accuracy": acc, "macro_f1": macro, "per_class_f1": f1, "mae": mae, "rmse": rmse, "pearson": pearson}
    expected = summary["metrics"]
    comparisons = {name: abs(value - expected[name]) <= 1e-10 for name, value in metrics.items() if name not in ("per_class_f1",)}
    comparisons["per_class_f1"] = all(abs(f1[name] - expected["per_class_f1"][name]) <= 1e-10 for name in CLASSES)
    csv_hash = sha256(pred_path)
    passed = (
        n == summary["sample_count"] == 727
        and ids_unique
        and csv_hash == EXPECTED_CSV_SHA256
        and manifest_ok
        and summary["confusion_matrix"]["rows_true_cols_pred"] == cm
        and not prob_failures
        and not finite_failures
        and all(comparisons.values())
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "inference_run": False,
        "row_count": n,
        "unique_sample_ids": len(set(ids)),
        "sample_ids_unique": ids_unique,
        "predictions_csv_sha256": csv_hash,
        "expected_predictions_csv_sha256": EXPECTED_CSV_SHA256,
        "output_manifest_entries": len(manifest),
        "output_manifest_verified": manifest_ok,
        "metrics_recomputed": metrics,
        "metric_matches_summary": comparisons,
        "confusion_matrix_recomputed": cm,
        "probability_simplex_failures": len(prob_failures),
        "nonfinite_intensity_rows": len(finite_failures),
        "matches_summary_sample_count": n == summary["sample_count"],
        "matches_summary_confusion_matrix": cm == summary["confusion_matrix"]["rows_true_cols_pred"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
