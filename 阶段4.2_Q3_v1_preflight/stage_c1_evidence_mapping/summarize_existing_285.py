"""Join the prior 285 index_only rows to the new read-only mapping probes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "reports" / "representative_evidence_mapping.csv"
TOKEN_ROWS = ROOT / "results" / "text_tokenizer_replay_20.csv"
OUT = ROOT / "results"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    previous = read_csv(PRIOR)
    token = read_csv(TOKEN_ROWS)
    token_by_key = {(row["sample_id"], row["official_seq_index"]): row for row in token}
    rows = []
    for item in previous:
        modality = item["modality"]
        token_record = token_by_key.get((item["sample_id"], item["official_seq_index"]))
        exact_offset = bool(token_record and token_record["ids_exact"] == "True" and
                            token_record["attention_exact"] == "True" and token_record["segments_exact"] == "True" and
                            int(token_record["char_end"]) > int(token_record["char_start"]))
        if modality == "text":
            residual = "raw_text_to_text_bert_token_offset_candidate_available; continuous_text_feature_row_provenance_not_verified"
        elif modality == "audio":
            residual = "PKL_has_no_explicit_per_row_audio_timestamp_or_feature_support_metadata; raw_video_audio_is_present; upstream recovery_not_assessed"
        else:
            residual = "PKL_has_no_explicit_per_row_video_PTS_or_feature_support_metadata; raw_video_is_present; upstream recovery_not_assessed"
        rows.append({
            "sample_id": item["sample_id"], "modality": modality,
            "official_seq_index": item["official_seq_index"],
            "prior_mapping_status": item["mapping_status"],
            "current_script_immediate_cause": "B_PRECHECK_UNCONDITIONAL_FAIL_CLOSED_INDEX_ONLY_FALLBACK",
            "tokenizer_char_offset_candidate_exact": exact_offset if modality == "text" else "NA",
            "candidate_char_start": token_record["char_start"] if modality == "text" and token_record else "",
            "candidate_char_end": token_record["char_end"] if modality == "text" and token_record else "",
            "candidate_token": token_record["token"] if modality == "text" and token_record else "",
            "candidate_raw_text_substring": token_record["raw_text_substring"] if modality == "text" and token_record else "",
            "verified_official_feature_row_mapping": False,
            "remaining_A_source_provenance_gap": residual,
        })
    if len(rows) != 285 or any(row["prior_mapping_status"] != "index_only" for row in rows):
        raise ValueError("expected the complete prior 285-row index_only set")
    path = OUT / "prior_285_root_cause.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for modality in ("text", "audio", "vision"):
        subset = [r for r in rows if r["modality"] == modality]
        counts[modality] = {
            "rows": len(subset),
            "prior_index_only": sum(r["prior_mapping_status"] == "index_only" for r in subset),
            "unconditional_implementation_fallback": len(subset),
            "token_offset_candidate_exact": sum(r["tokenizer_char_offset_candidate_exact"] is True for r in subset),
            "verified_official_feature_row_mapping": sum(bool(r["verified_official_feature_row_mapping"]) for r in subset),
        }
    summary = {
        "text_mapping": "BLOCKED",
        "audio_mapping": "BLOCKED",
        "vision_mapping": "BLOCKED",
        "root_cause_classification": {
            "immediate_285_row_cause": "B_PRECHECK_PATH_UNCONDITIONALLY_EMITTED_INDEX_ONLY; NOT_285_OBSERVED_MAPPING_FAILURES",
            "residual_evidence_limit": "A_CURRENT_PKLS_OMIT_EXPLICIT_PER_ROW_SOURCE_METADATA; UPSTREAM_RECOVERY_NOT_ASSESSED",
            "global_data_impossibility_proven": False,
        },
        "prior_285_rows": len(rows),
        "by_modality": counts,
        "text_subcheck": "raw_text_to_text_bert token offsets replay exactly for all 20 Attachment4 records; this does not verify continuous text feature row provenance",
        "audio_vision_subcheck": "PKLs contain no obvious per-row time/PTS/frame fields; raw video pairs exist; the prior precheck did not attempt temporal reconstruction or official-row support verification",
        "model_changed": False,
        "training_run": False,
        "attachment4_prediction_run": False,
    }
    (OUT / "root_cause_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
