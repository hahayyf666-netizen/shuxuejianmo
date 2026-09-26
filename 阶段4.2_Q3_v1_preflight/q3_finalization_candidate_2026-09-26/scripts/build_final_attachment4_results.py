from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

from common import CANDIDATE, MODALITIES, RESULTS, STAGE, max_absolute, max_positive, phi_values, read_csv, read_json, top_indices, write_csv, write_json

HIST_RESULTS = STAGE / "external_review_handoff_2026-09-26/artifacts_extracted/formal_attachment4_2026-09-26/results"
T4_RESULTS = STAGE / "t4_raw_evidence_closure_2026-09-26/results"
T4_VISUAL_OK = "reconstructed_keyframe_from_verified_lineage"
T4_AUDIO_OK = "reconstructed_speech_interval_from_verified_lineage"
ANOMALIES = {
    "06": "T2 vision row nonunique",
    "13": "aligned vision all-zero / C2 visual chain unavailable",
    "16": "aligned vision all-zero-like / C2 visual chain unavailable",
    "18": "T2 vision row nonunique",
}


def compact_evidence(records: list[dict], modality: str) -> str:
    """Deduplicate repeated raw evidence units while preserving feature indices."""
    grouped: dict[str, dict] = {}
    for row in records:
        key = row.get("evidence_unit_id") or f"{row.get('local_start_sec')}:{row.get('local_end_sec')}:{row.get('frame_path')}"
        if key not in grouped:
            grouped[key] = dict(row)
            grouped[key]["official_seq_indices"] = [str(row.get("official_seq_index", ""))]
            grouped[key]["attribution_targets"] = list(row.get("attribution_targets", []))
        else:
            idx = str(row.get("official_seq_index", ""))
            if idx not in grouped[key]["official_seq_indices"]:
                grouped[key]["official_seq_indices"].append(idx)
            grouped[key]["attribution_targets"] = sorted(set(grouped[key].get("attribution_targets", []) + list(row.get("attribution_targets", []))))
    out = []
    for item in grouped.values():
        if modality == "audio":
            out.append({
                "official_seq_indices": ";".join(item["official_seq_indices"]),
                "local_interval_sec": [float(item["local_start_sec"]), float(item["local_end_sec"])],
                "attribution_targets": sorted(set(item.get("attribution_targets", []))),
                "mapping_status": item["mapping_status"],
            })
        else:
            out.append({
                "official_seq_indices": ";".join(item["official_seq_indices"]),
                "local_pts_sec": float(item["selected_local_pts"]),
                "frame_path": f"{STAGE.relative_to(STAGE.parent).as_posix()}/t4_raw_evidence_closure_2026-09-26/{item['frame_path']}",
                "evidence_unit_id": item.get("evidence_unit_id"),
                "attribution_targets": sorted(set(item.get("attribution_targets", []))),
                "mapping_status": item["mapping_status"],
            })
    return json.dumps(out, ensure_ascii=False, separators=(",", ":")) if out else "feature_position_only"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historical-results", type=Path, default=HIST_RESULTS)
    ap.add_argument("--t4-results", type=Path, default=T4_RESULTS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_csv = args.historical_results / "predictions_explanations.csv"
    sample_dir = args.historical_results / "samples"
    visual_rows = read_csv(args.t4_results / "t4_visual_evidence.csv")
    audio_rows = read_csv(args.t4_results / "t4_audio_evidence.csv")
    t4_gate = read_json(args.t4_results / "T4_GATE.json")
    if t4_gate.get("gate") != "T4_COMPLETE_WITH_LIMITATIONS" or t4_gate.get("historical_4_6_gate_modified") or t4_gate.get("4_7_entered"):
        raise RuntimeError("T4 gate/scope differs from frozen status")
    hist_rows = read_csv(source_csv)
    if len(hist_rows) != 20 or {r["sample_id"] for r in hist_rows} != {f"{i:02d}" for i in range(1, 21)}:
        raise RuntimeError("frozen Attachment4 CSV must contain exactly samples 01-20")
    samples = {sid: read_json(sample_dir / f"sample_{sid}.json") for sid in (f"{i:02d}" for i in range(1, 21))}
    if any(samples[sid].get("prediction", {}).get("class_name") != next(r["predicted_class"] for r in hist_rows if r["sample_id"] == sid) for sid in samples):
        raise RuntimeError("historical per-sample JSON and CSV prediction classes differ")

    evidence = defaultdict(lambda: {"audio": [], "vision": []})
    for row in audio_rows:
        if row.get("mapping_status") == T4_AUDIO_OK:
            evidence[(row["sample_id"], int(row["official_seq_index"]))]["audio"].append(row)
    for row in visual_rows:
        if row.get("mapping_status") == T4_VISUAL_OK:
            evidence[(row["sample_id"], int(row["official_seq_index"]))]["vision"].append(row)
    audio_good = [r for r in audio_rows if r.get("mapping_status") == T4_AUDIO_OK]
    visual_good = [r for r in visual_rows if r.get("mapping_status") == T4_VISUAL_OK]
    coverage_rows = []
    final_rows = []
    anomaly_rows = []
    primary_cov = dominant_cls_cov = dominant_reg_cov = 0

    def mapped(sid: str, modality: str, positions: list[int]) -> tuple[bool, list[dict]]:
        if modality == "text":
            rec = samples[sid]
            target_mod = rec["targets"]
            candidates = []
            for target in ("classification", "regression"):
                for p in target_mod[target]["modalities"]["text"]["top_positions"]:
                    if int(p["official_seq_index"]) in positions and p.get("mapping_status") == "verified_text" and int(p.get("char_end", -1)) > int(p.get("char_start", -1)) and p.get("raw_text_substring"):
                        candidates.append(p)
            return bool(candidates), candidates
        kind = "audio" if modality == "audio" else "vision"
        recs = []
        for idx in positions:
            recs.extend(evidence[(sid, idx)][kind])
        return bool(recs), recs

    for row in hist_rows:
        sid = row["sample_id"]
        rec = samples[sid]
        cls_phi = phi_values(row, "classification")
        reg_phi = phi_values(row, "regression")
        primary = max_positive(cls_phi)
        if primary is None:
            raise RuntimeError(f"{sid}: classification has no positive Shapley modality; primary reference undefined")
        dominant_cls = max_absolute(cls_phi)
        dominant_reg = max_absolute(reg_phi)
        if row.get("primary_supporting_modality_cls") != primary:
            raise RuntimeError(f"{sid}: primary reference recomputation disagrees with frozen field")
        if row.get("primary_influential_modality_cls") != dominant_cls or row.get("primary_influential_modality_reg") != dominant_reg:
            raise RuntimeError(f"{sid}: dominant modality recomputation disagrees with frozen fields")

        chosen = {}
        for name, modality, target in [
            ("primary", primary, "classification"),
            ("dominant_cls", dominant_cls, "classification"),
            ("dominant_reg", dominant_reg, "regression"),
        ]:
            ok, mapped_records = mapped(sid, modality, top_indices(row, target, modality))
            chosen[name] = (ok, mapped_records)
        primary_cov += int(chosen["primary"][0])
        dominant_cls_cov += int(chosen["dominant_cls"][0])
        dominant_reg_cov += int(chosen["dominant_reg"][0])

        mod_status = {}
        mod_records = {}
        for modality in MODALITIES:
            positions = set(top_indices(row, "classification", modality) + top_indices(row, "regression", modality))
            if modality == "text":
                ok, items = mapped(sid, modality, list(positions))
                mod_status[modality] = "verified_text" if ok else "text_mapping_unavailable"
                mod_records[modality] = items
            else:
                ok, items = mapped(sid, modality, sorted(positions))
                mod_status[modality] = "verified_time" if ok and modality == "audio" else ("verified_time" if ok else "feature_position_only") if modality == "audio" else ("verified_time" if ok else "feature_position_only")
                # Vision is a frame mapping, not a time-only assertion.
                if modality == "vision" and ok:
                    mod_status[modality] = "reconstructed_keyframe_from_verified_lineage"
                mod_records[modality] = items

        coverage_rows.append({
            "sample_id": sid,
            "primary_reference_modality_cls": primary,
            "primary_reference_raw_evidence_available": int(chosen["primary"][0]),
            "dominant_influence_modality_cls": dominant_cls,
            "dominant_cls_raw_evidence_available": int(chosen["dominant_cls"][0]),
            "dominant_influence_modality_reg": dominant_reg,
            "dominant_reg_raw_evidence_available": int(chosen["dominant_reg"][0]),
            "text_evidence_status": mod_status["text"],
            "audio_evidence_status": mod_status["audio"],
            "vision_evidence_status": mod_status["vision"],
            "known_input_anomaly": row.get("known_input_anomaly") or ANOMALIES.get(sid, ""),
        })

        target_texts = {}
        for target, prefix in [("classification", "cls"), ("regression", "reg")]:
            vals = []
            for p in rec["targets"][target]["modalities"]["text"]["top_positions"]:
                vals.append({"official_seq_index": p["official_seq_index"], "char_span": [p.get("char_start"), p.get("char_end")], "text": p.get("raw_text_substring"), "signed_importance": p["signed_importance"], "status": p["mapping_status"]})
            target_texts[target] = vals
        text_evidence = json.dumps(target_texts, ensure_ascii=False, separators=(",", ":"))
        relevant_audio = []
        relevant_vision = []
        for target, modality in [("classification", "audio"), ("regression", "audio")]:
            _, items = mapped(sid, modality, top_indices(row, target, modality))
            relevant_audio.extend({**item, "attribution_targets": [target]} for item in items)
        for target, modality in [("classification", "vision"), ("regression", "vision")]:
            _, items = mapped(sid, modality, top_indices(row, target, modality))
            relevant_vision.extend({**item, "attribution_targets": [target]} for item in items)
        final_rows.append({
            "sample_id": sid, "official_id": row["official_id"], "predicted_class": row["predicted_class"],
            "p_negative": row["p_negative"], "p_neutral": row["p_neutral"], "p_positive": row["p_positive"], "predicted_intensity": row["predicted_intensity"],
            **{k: row[k] for k in ("phi_cls_text", "phi_cls_audio", "phi_cls_vision", "phi_reg_text", "phi_reg_audio", "phi_reg_vision")},
            "primary_reference_modality_cls": primary, "dominant_influence_modality_cls": dominant_cls,
            "dominant_influence_modality_reg": dominant_reg,
            "dominant_reg_direction": "push_positive" if reg_phi[dominant_reg] > 1e-6 else "push_negative" if reg_phi[dominant_reg] < -1e-6 else "neutral_relative_to_reference",
            **{k: row[k] for k in ("top_seq_cls_text", "top_seq_cls_audio", "top_seq_cls_vision", "top_seq_reg_text", "top_seq_reg_audio", "top_seq_reg_vision", "top_char_spans_cls_text", "top_char_spans_reg_text")},
            "text_evidence_status": mod_status["text"], "audio_evidence_status": mod_status["audio"], "vision_evidence_status": mod_status["vision"],
            "key_text_evidence": text_evidence,
            "key_audio_evidence": compact_evidence(relevant_audio, "audio"),
            "key_visual_evidence": compact_evidence(relevant_vision, "vision"),
            "known_input_anomaly": row.get("known_input_anomaly") or ANOMALIES.get(sid, ""),
            "interpretation_note": ("Vision attribution is feature-branch attribution only; raw visual evidence unavailable." if sid in ("13", "16") else "T2 visual provenance is ambiguous; do not claim raw visual evidence." if sid in ("06", "18") else "Signed Shapley/IG describe frozen-model feature-space response; raw media evidence is limited to verified mappings."),
        })

        if sid in ANOMALIES:
            anomaly_rows.append({
                "sample_id": sid, "known_input_anomaly": ANOMALIES[sid],
                "aligned_vision_all_zero": int(sid == "13"), "aligned_vision_all_zero_like": int(sid == "16"),
                "vision_evidence_level": "feature_position_only / index_only",
                "raw_visual_evidence_available": int(bool(mod_records["vision"])),
                "interpretation_note": "Preserve prediction and attributions; no visual keyframe is inferred from the anomaly." if sid in ("13", "16") else "Preserve prediction and attributions; T2 nonunique visual provenance does not establish a raw frame.",
            })

    write_csv(out / "final_evidence_coverage_20.csv", coverage_rows, list(coverage_rows[0]))
    write_csv(out / "attachment4_predictions_explanations_final.csv", final_rows, list(final_rows[0]))
    write_csv(out / "attachment4_anomaly_summary.csv", anomaly_rows, list(anomaly_rows[0]))
    tallies = {
        "sample_count": len(coverage_rows),
        "primary_reference_raw_evidence_coverage": sum(r["primary_reference_raw_evidence_available"] for r in coverage_rows),
        "dominant_cls_raw_evidence_coverage": sum(r["dominant_cls_raw_evidence_available"] for r in coverage_rows),
        "dominant_reg_raw_evidence_coverage": sum(r["dominant_reg_raw_evidence_available"] for r in coverage_rows),
        "expected_reference_counts": {"primary_reference": 20, "dominant_cls": 20, "dominant_reg": 19},
        "matches_expected_reference_counts": [sum(r["primary_reference_raw_evidence_available"] for r in coverage_rows), sum(r["dominant_cls_raw_evidence_available"] for r in coverage_rows), sum(r["dominant_reg_raw_evidence_available"] for r in coverage_rows)] == [20, 20, 19],
        "sample05_vision": next(r["vision_evidence_status"] for r in coverage_rows if r["sample_id"] == "05"),
        "sample14_visual_same_evidence_unit_deduplicated": len({r.get("evidence_unit_id") for r in visual_rows if r["sample_id"] == "14" and r.get("mapping_status") == T4_VISUAL_OK and r.get("evidence_unit_id")}) < sum(1 for r in visual_rows if r["sample_id"] == "14" and r.get("mapping_status") == T4_VISUAL_OK),
        "t4_gate": t4_gate["gate"],
    }
    write_json(out / "final_evidence_coverage_summary.json", tallies)
    print(json.dumps(tallies, ensure_ascii=False))
    if not tallies["matches_expected_reference_counts"] or tallies["sample05_vision"] != "feature_position_only":
        raise SystemExit("STOP_REVIEW: recomputed evidence coverage differs from frozen expectation")


if __name__ == "__main__":
    main()
