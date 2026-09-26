from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from common import CLASS_NAMES, REPO, RESULTS, read_csv, write_csv, write_json



def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return resolved.name

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-predictions", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    manifest_path = args.source_manifest or (args.test_predictions.parent / "output_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get(args.test_predictions.name)
    actual_sha = hashlib.sha256(args.test_predictions.read_bytes()).hexdigest()
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError("historical test_predictions.csv does not match the frozen output manifest")
    rows = read_csv(args.test_predictions)
    if len(rows) != 727 or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("historical test prediction file must contain 727 unique sample IDs")
    conflicts = []
    margins = []
    neutral = []
    for r in rows:
        probs = [float(r[f"p_{label.lower()}"]) for label in CLASS_NAMES]
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])
        intensity = float(r["predicted_intensity"])
        row = {"sample_id": r["sample_id"], "predicted_class": r["predicted_class"], "p_negative": probs[0], "p_neutral": probs[1], "p_positive": probs[2], "classification_margin": margin, "predicted_intensity": intensity, "absolute_predicted_intensity": abs(intensity)}
        margins.append(margin)
        if r["predicted_class"] == "Neutral":
            neutral.append(row)
        if (r["predicted_class"] == "Negative" and intensity > 0) or (r["predicted_class"] == "Positive" and intensity < 0):
            conflicts.append({**row, "low_intensity_abs_lt_0_2": int(abs(intensity) < 0.2), "low_margin_lt_0_2": int(margin < 0.2)})
    observed = {"sample_count": len(rows), "direct_opposite_polarity_conflicts": len(conflicts), "direct_conflicts_abs_intensity_lt_0_2": sum(r["low_intensity_abs_lt_0_2"] for r in conflicts), "direct_conflicts_margin_lt_0_2": sum(r["low_margin_lt_0_2"] for r in conflicts), "predicted_neutral_count": len(neutral), "predicted_neutral_abs_intensity_ge_0_2_count": sum(r["absolute_predicted_intensity"] >= 0.2 for r in neutral)}
    expected = {"sample_count": 727, "direct_opposite_polarity_conflicts": 22, "direct_conflicts_abs_intensity_lt_0_2": 21, "direct_conflicts_margin_lt_0_2": 20, "predicted_neutral_count": 91, "predicted_neutral_abs_intensity_ge_0_2_count": 14}
    match = observed == expected
    write_csv(args.out / "test_direct_conflicts.csv", conflicts, list(conflicts[0]) if conflicts else ["sample_id"])
    write_json(args.out / "test_consistency_summary.json", {
        "status": "PASS" if match else "STOP_REVIEW_EXPECTED_COUNTS_DIFFER",
        "source": display_path(args.test_predictions), "source_manifest": display_path(manifest_path), "source_manifest_sha256_for_test_predictions": expected_sha,
        "source_manifest_match": actual_sha == expected_sha, "observed": observed, "expected_reference_counts_not_used_for_tuning": expected,
        "matches_expected_reference_counts": match,
        "classification_margin_distribution": {"count": len(margins), "mean": float(np.mean(margins)), "median": float(np.median(margins)), "p10": float(np.quantile(margins, .1)), "p25": float(np.quantile(margins, .25)), "p75": float(np.quantile(margins, .75)), "p90": float(np.quantile(margins, .9))},
        "analysis_only": True, "test_used_for_tuning": False, "prediction_changed": False,
    })
    print({"status": "PASS" if match else "STOP_REVIEW_EXPECTED_COUNTS_DIFFER", **observed})
    if not match:
        raise SystemExit("STOP: test consistency counts differ from the stated frozen reference counts; do not tune")


if __name__ == "__main__":
    main()
