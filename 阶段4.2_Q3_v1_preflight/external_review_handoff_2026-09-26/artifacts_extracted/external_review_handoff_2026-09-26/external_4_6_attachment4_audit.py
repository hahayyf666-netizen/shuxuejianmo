"""Read-only audit of frozen Attachment4 result artifacts; never loads a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


IDS = [f"{i:02d}" for i in range(1, 21)]
MODALITIES = ("text", "audio", "vision")
TARGETS = ("classification", "regression")
FORBIDDEN_KEYS = ("time", "pts", "frame", "second", "causal")
FORBIDDEN_CLAIMS = ("causal contribution", "causal effect", "seconds", "pts", "frame time")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key_has_forbidden_temporal_claim(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(word in normalized for word in FORBIDDEN_KEYS):
                return True
            if key_has_forbidden_temporal_claim(item):
                return True
    elif isinstance(value, list):
        return any(key_has_forbidden_temporal_claim(item) for item in value)
    elif isinstance(value, str):
        low = value.lower()
        return any(claim in low for claim in FORBIDDEN_CLAIMS)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Frozen Attachment4 results/ directory")
    parser.add_argument("--out", type=Path, required=True, help="New audit output directory")
    args = parser.parse_args()
    root = args.results.resolve()
    out = args.out.resolve()
    if not root.is_dir():
        raise SystemExit(f"results directory not found: {root}")
    if out == root or root in out.parents:
        raise SystemExit("audit output must be outside the frozen results directory")
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "output_manifest.json").read_text(encoding="utf-8"))
    manifest_ok = all((root / rel).is_file() and sha256(root / rel) == expected for rel, expected in manifest.items())
    rows = []
    samples = {}
    for sid in IDS:
        path = root / "samples" / f"sample_{sid}.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        samples[sid] = rec
        probs = rec["prediction"]["probabilities"]
        prob_sum = sum(float(probs[c]) for c in ("Negative", "Neutral", "Positive"))
        intensity = float(rec["prediction"]["intensity"])
        cls = rec["targets"]["classification"]
        reg = rec["targets"]["regression"]
        content = set(int(x) for x in rec["official_seq_indices"])
        all_text_spans_valid = True
        all_top_positions_in_content = True
        modality_time_clean = True
        target_complete = {}
        for target, result in (("classification", cls), ("regression", reg)):
            coal = result["shapley"]["coalition_values"]
            phi = result["shapley"]["phi"]
            shapley_resid = sum(float(phi[m]) for m in MODALITIES) - (float(coal["7"]) - float(coal["0"]))
            shapley_stored = float(result["shapley"]["additivity_residual"])
            top_text = result["modalities"]["text"]["top_positions"]
            text_ok = bool(top_text)
            for pos in top_text:
                start, end = pos.get("char_start"), pos.get("char_end")
                token_ok = pos.get("token_id") is not None
                span_ok = start is not None and end is not None and 0 <= int(start) < int(end) <= len(rec["raw_text"])
                sub_ok = span_ok and rec["raw_text"][int(start):int(end)] == pos.get("raw_text_substring")
                text_ok = text_ok and token_ok and sub_ok and pos.get("mapping_status") == "verified_text"
                all_text_spans_valid = all_text_spans_valid and token_ok and sub_ok
            for modality in MODALITIES:
                m = result["modalities"][modality]
                for pos in m["top_positions"]:
                    all_top_positions_in_content = all_top_positions_in_content and int(pos["official_seq_index"]) in content
                if modality in ("audio", "vision"):
                    modality_time_clean = modality_time_clean and not key_has_forbidden_temporal_claim(m["top_positions"])
            ig_ok = True
            ig_stats = {}
            for modality in MODALITIES:
                ig = result["modalities"][modality]["conditional_ig"]
                residual = float(ig["completeness_residual"])
                delta = abs(float(ig["conditional_output_difference"]))
                tolerance = 1e-3 + 0.01 * delta
                passed = ig["numerical_status"] == "pass" and abs(residual) <= tolerance
                ig_ok = ig_ok and passed
                ig_stats[modality] = {"residual": residual, "tolerance": tolerance, "pass": passed}
            target_complete[target] = {
                "shapley_residual_recomputed": shapley_resid,
                "shapley_residual_stored": shapley_stored,
                "shapley_pass": abs(shapley_resid) <= 1e-6 and abs(shapley_resid - shapley_stored) <= 1e-6,
                "ig": ig_stats,
                "ig_all_pass": ig_ok,
                "primary_influential_modality": result["semantics"].get("primary_influential_modality"),
                "primary_supporting_modality": result["semantics"].get("primary_supporting_modality"),
                "text_top_evidence_valid": text_ok,
            }
        rows.append({
            "sample_id": sid,
            "classification_primary_modality": cls["semantics"].get("primary_influential_modality"),
            "regression_primary_modality": reg["semantics"].get("primary_influential_modality"),
            "text_raw_evidence_available": all_text_spans_valid,
            "audio_raw_evidence_available": False,
            "vision_raw_evidence_available": False,
            "classification_key_evidence_requirement_satisfied": cls["semantics"].get("primary_influential_modality") == "text" and target_complete["classification"]["text_top_evidence_valid"],
            "regression_key_evidence_requirement_satisfied": reg["semantics"].get("primary_influential_modality") == "text" and target_complete["regression"]["text_top_evidence_valid"],
            "reason": "text primary with verified raw span" if cls["semantics"].get("primary_influential_modality") == "text" or reg["semantics"].get("primary_influential_modality") == "text" else "primary modality is audio/vision and remains index_only",
            "mapping_status_text": rec["mapping_status"]["text"],
            "mapping_status_audio": rec["mapping_status"]["audio"],
            "mapping_status_vision": rec["mapping_status"]["vision"],
            "known_input_anomaly": rec.get("known_input_anomaly") or "",
            "probability_sum": prob_sum,
            "probability_pass": all(math.isfinite(float(probs[c])) for c in probs) and abs(prob_sum - 1.0) <= 1e-6,
            "predicted_intensity": intensity,
            "intensity_pass": math.isfinite(intensity) and -3.0 <= intensity <= 3.0,
            "padding_top_position_check_pass": all_top_positions_in_content,
            "audio_vision_temporal_claim_check_pass": modality_time_clean,
            "targets": target_complete,
        })

    expected = set(IDS)
    actual = set(samples)
    required_anomalies = {"06", "13", "16", "18"}
    influence_cls = Counter(r["classification_primary_modality"] for r in rows)
    support_cls = Counter(samples[s]["targets"]["classification"]["semantics"].get("primary_supporting_modality") for s in samples)
    influence_reg = Counter(r["regression_primary_modality"] for r in rows)
    # The frozen result schema has no regression primary_supporting_modality.
    # Do not invent that semantic; export its absence explicitly.
    result = {
        "status": "PASS" if actual == expected and manifest_ok and all(r["probability_pass"] and r["intensity_pass"] and r["padding_top_position_check_pass"] and r["audio_vision_temporal_claim_check_pass"] and all(t["shapley_pass"] and t["ig_all_pass"] for t in r["targets"].values()) for r in rows) and required_anomalies.issubset({s for s, x in samples.items() if x.get("known_input_anomaly")}) else "FAIL",
        "sample_ids_exact_01_20": actual == expected,
        "sample_count": len(rows),
        "manifest_entries": len(manifest),
        "manifest_sha_verified": manifest_ok,
        "probability_simplex_pass_count": sum(bool(r["probability_pass"]) for r in rows),
        "intensity_range_pass_count": sum(bool(r["intensity_pass"]) for r in rows),
        "shapley_additivity_checks": sum(1 for r in rows for t in r["targets"].values()),
        "shapley_additivity_pass_count": sum(1 for r in rows for t in r["targets"].values() if t["shapley_pass"]),
        "conditional_ig_checks": sum(1 for r in rows for t in r["targets"].values() for m in MODALITIES),
        "conditional_ig_pass_count": sum(1 for r in rows for t in r["targets"].values() for m in MODALITIES if t["ig"][m]["pass"]),
        "padding_top_positions_pass_count": sum(bool(r["padding_top_position_check_pass"]) for r in rows),
        "audio_vision_time_claim_checks_pass_count": sum(bool(r["audio_vision_temporal_claim_check_pass"]) for r in rows),
        "mapping_status_counts": {m: dict(Counter(samples[s]["mapping_status"][m] for s in samples)) for m in MODALITIES},
        "known_anomaly_samples_present": sorted(required_anomalies.intersection({s for s, x in samples.items() if x.get("known_input_anomaly")})),
        "classification_primary_influential_counts": dict(influence_cls),
        "classification_primary_supporting_counts": dict(support_cls),
        "regression_primary_influential_counts": dict(influence_reg),
        "regression_primary_supporting": "not_defined_in_frozen_result_schema; not inferred",
        "classification_audio_or_vision_primary_samples": [r["sample_id"] for r in rows if r["classification_primary_modality"] in ("audio", "vision")],
        "regression_audio_or_vision_primary_samples": [r["sample_id"] for r in rows if r["regression_primary_modality"] in ("audio", "vision")],
        "evidence_coverage": {
            "classification_satisfied_count": sum(bool(r["classification_key_evidence_requirement_satisfied"]) for r in rows),
            "regression_satisfied_count": sum(bool(r["regression_key_evidence_requirement_satisfied"]) for r in rows),
        },
        "row_audit": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "attachment4_external_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage_path = out / "primary_evidence_coverage_20.csv"
    fields = [k for k in rows[0] if k != "targets"]
    with coverage_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (dict, list)) else row[k] for k in fields})
    print(json.dumps({k: v for k, v in result.items() if k != "row_audit"}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
