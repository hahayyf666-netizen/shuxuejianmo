"""Export a public-safe, one-row-per-Attachment4 explanation table.

The source JSON contains official text for local verification.  This export
intentionally omits raw text and raw feature arrays; it keeps only predictions,
modality-level attributions, feature-position references, and mapping states.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODALITIES = ("text", "audio", "vision")


def _top_positions(record: dict, target: str, modality: str) -> list[dict]:
    return (
        record["targets"][target]["modalities"][modality]["top_positions"]
    )


def _seqs(positions: list[dict]) -> str:
    return ";".join(str(int(p["official_seq_index"])) for p in positions)


def _spans(positions: list[dict]) -> str:
    spans = []
    for p in positions:
        if p.get("char_start") is None or p.get("char_end") is None:
            continue
        spans.append(f"{int(p['char_start'])}:{int(p['char_end'])}")
    return ";".join(spans)


def export(source: Path, target: Path) -> None:
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError(f"expected 20 records, got {type(rows).__name__}/{len(rows)}")

    fields = [
        "sample_id",
        "official_id",
        "valid_content_length",
        "predicted_class",
        "p_negative",
        "p_neutral",
        "p_positive",
        "predicted_intensity",
        "phi_cls_text",
        "phi_cls_audio",
        "phi_cls_vision",
        "phi_reg_text",
        "phi_reg_audio",
        "phi_reg_vision",
        "primary_influential_modality_cls",
        "primary_supporting_modality_cls",
        "primary_influential_modality_reg",
        "mapping_status_text",
        "mapping_status_audio",
        "mapping_status_vision",
        "media_origin_audit",
        "known_input_anomaly",
        "local_status_cls_text",
        "local_status_cls_audio",
        "local_status_cls_vision",
        "local_status_reg_text",
        "local_status_reg_audio",
        "local_status_reg_vision",
        "top_seq_cls_text",
        "top_seq_cls_audio",
        "top_seq_cls_vision",
        "top_seq_reg_text",
        "top_seq_reg_audio",
        "top_seq_reg_vision",
        "top_char_spans_cls_text",
        "top_char_spans_reg_text",
    ]

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in sorted(rows, key=lambda r: r["sample_id"]):
            pred = record["prediction"]
            cls = record["targets"]["classification"]
            reg = record["targets"]["regression"]
            row = {
                "sample_id": record["sample_id"],
                "official_id": record["official_id"],
                "valid_content_length": record["valid_content_length"],
                "predicted_class": pred["class_name"],
                "p_negative": pred["probabilities"]["Negative"],
                "p_neutral": pred["probabilities"]["Neutral"],
                "p_positive": pred["probabilities"]["Positive"],
                "predicted_intensity": pred["intensity"],
                "phi_cls_text": cls["shapley"]["phi"]["text"],
                "phi_cls_audio": cls["shapley"]["phi"]["audio"],
                "phi_cls_vision": cls["shapley"]["phi"]["vision"],
                "phi_reg_text": reg["shapley"]["phi"]["text"],
                "phi_reg_audio": reg["shapley"]["phi"]["audio"],
                "phi_reg_vision": reg["shapley"]["phi"]["vision"],
                "primary_influential_modality_cls": cls["semantics"]["primary_influential_modality"],
                "primary_supporting_modality_cls": cls["semantics"]["primary_supporting_modality"],
                "primary_influential_modality_reg": reg["semantics"]["primary_influential_modality"],
                "mapping_status_text": record["mapping_status"]["text"],
                "mapping_status_audio": record["mapping_status"]["audio"],
                "mapping_status_vision": record["mapping_status"]["vision"],
                "media_origin_audit": record.get("media_origin_audit"),
                "known_input_anomaly": record.get("known_input_anomaly") or "",
            }
            for target_name, target_record, prefix in (
                ("classification", cls, "cls"),
                ("regression", reg, "reg"),
            ):
                for modality in MODALITIES:
                    positions = _top_positions(record, target_name, modality)
                    row[f"local_status_{prefix}_{modality}"] = target_record["modalities"][modality]["local_attribution_status"]
                    row[f"top_seq_{prefix}_{modality}"] = _seqs(positions)
                row[f"top_char_spans_{prefix}_text"] = _spans(_top_positions(record, target_name, "text"))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)
    print(json.dumps({"status": "EXPORTED", "rows": 20, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
