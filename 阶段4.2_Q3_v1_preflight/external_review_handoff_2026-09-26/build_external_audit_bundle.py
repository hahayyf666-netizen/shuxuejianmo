"""Package existing Q3 audit evidence for a separately assigned reviewer.

This script copies only existing files into a ZIP. It never runs model code or
changes the source artifacts. The original aligned PKL and source videos are
not included; their hashes and access limitations are documented separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True, help="Workspace containing work/q3-xai-freeze and outputs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ws = args.workspace.resolve()
    stage = ws / "work" / "q3-xai-freeze" / "阶段4.2_Q3_v1_preflight"
    results = stage / "formal_attachment4_2026-09-26" / "results"
    if not results.is_dir():
        raise SystemExit(f"Q3 stage not found: {stage}")
    sources: dict[Path, str] = {}

    def add(path: Path, archive_name: str | None = None) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = archive_name or path.relative_to(stage).as_posix()
        sources[path] = rel

    # Original formal output trees and test artifacts, copied byte-for-byte.
    for path in results.rglob("*"):
        if path.is_file():
            add(path)
    for path in (stage / "formal_test_evaluation_2026-09-26").glob("*"):
        if path.is_file():
            add(path)
    for name in ("B0_seed2029.pt", "train_scaler.pt", "attachment4_explanation_contract.json"):
        add(stage / "formal_attachment4_2026-09-26" / "assets" / name)
    for rel in (
        "formal_attachment4_2026-09-26/run_attachment4_formal.py",
        "formal_attachment4_2026-09-26/run_internal_acceptance.py",
        "formal_attachment4_2026-09-26/freeze_attachment4_contract.py",
        "formal_attachment4_2026-09-26/export_predictions_csv.py",
        "formal_attachment4_2026-09-26/export_attachment4_csv_from_frozen_json.py",
        "formal_attachment4_2026-09-26/finalize_4_5.py",
        "formal_attachment4_2026-09-26/final_4_5_review/Q3_4_5_FINAL_REVIEW_REPORT.md",
        "formal_attachment4_2026-09-26/final_4_5_review/q3_4_5_final_review.json",
        "formal_attachment4_2026-09-26/final_4_5_review/metadata_erratum.json",
        "formal_attachment4_2026-09-26/independent_4_6_review/Q3_4_6_INDEPENDENT_REVIEW_REPORT.md",
        "formal_attachment4_2026-09-26/independent_4_6_review/q3_4_6_gate.json",
        "formal_attachment4_2026-09-26/Q3_4_5_INTERNAL_ACCEPTANCE_REPORT.md",
        "formal_attachment4_2026-09-26/internal_acceptance.json",
        "formal_attachment4_2026-09-26/SHA256SUMS.txt",
        "formal_training_audit_drx_cuda128_2026-09-26/FORMAL_TRAINING_AUDIT_REPORT.md",
        "formal_training_audit_drx_cuda128_2026-09-26/MODEL_FREEZE_MANIFEST.json",
        "formal_training_audit_drx_cuda128_2026-09-26/training_summary.json",
        "formal_training_audit_drx_cuda128_2026-09-26/audit_result.json",
        "formal_training_audit_drx_cuda128_2026-09-26/SHA256SUMS.txt",
        "formal_xai_review_2026-09-26/Q3_VALID120_XAI_REVIEW_REPORT.md",
        "formal_xai_review_2026-09-26/REVIEW_GATE.json",
        "formal_xai_validation_2026-09-26/run_formal_xai_validation.py",
        "formal_xai_validation_2026-09-26/validation_contract.json",
        "formal_xai_validation_2026-09-26/assets/explanation_scope_contract.json",
        "formal_xai_validation_2026-09-26/assets/MODEL_FREEZE_MANIFEST.json",
        "run_formal_training.py",
        "run_final_test_evaluation.py",
        "frozen_config.json",
        "q3v1/model.py",
        "q3v1/data.py",
        "q3v1/train_eval.py",
        "q3v1/explain.py",
        "q3v1/mapping.py",
        "q3v1/scope_gate.py",
        "stage_c2_provenance/c2_gate.json",
        "stage_c2_provenance/results/aligned_to_unaligned_row_matches_20.csv",
        "t2_row_boundary_and_source_inventory_2026-09-26/T2_ROW_LINEAGE_AND_MEDIA_INVENTORY_REPORT.md",
        "t2_row_boundary_and_source_inventory_2026-09-26/results/t2_aggregate_gate.json",
        "t2_row_boundary_and_source_inventory_2026-09-26/results/t2_frozen_rules.json",
        "t2_row_boundary_and_source_inventory_2026-09-26/results/aligned_position_lineage_20.csv",
        "t3_media_origin_2026-09-26/T3_MEDIA_ORIGIN_REPORT.md",
        "t3_media_origin_2026-09-26/run_t3_media_origin.py",
        "t3_media_origin_2026-09-26/results_v3/t3_aggregate_gate.json",
        "t3_media_origin_2026-09-26/results_v3/t3_frozen_contract.json",
        "t3_media_origin_2026-09-26/results_v3/t3_media_origin_records.json",
        "external_review_handoff_2026-09-26/README.md",
        "external_review_handoff_2026-09-26/EXTERNAL_AUDIT_PREPARATION_REPORT.md",
        "external_review_handoff_2026-09-26/build_external_audit_bundle.py",
        "external_review_handoff_2026-09-26/external_4_6_attachment4_audit.py",
        "external_review_handoff_2026-09-26/external_4_6_test_audit.py",
        "external_review_handoff_2026-09-26/FORMAL_RUN_SOURCE_PROVENANCE.json",
        "external_review_handoff_2026-09-26/test_audit_result.json",
        "external_review_handoff_2026-09-26/attachment4_audit_results/attachment4_audit_aggregate.json",
        "external_review_handoff_2026-09-26/attachment4_audit_results/attachment4_external_audit.json",
        "external_review_handoff_2026-09-26/attachment4_audit_results/primary_evidence_coverage_20.csv",
    ):
        add(stage / rel)
    for path in sorted((stage / "t3_media_origin_2026-09-26" / "results_v3" / "per_sample").glob("*.json")):
        add(path)
    xai_zip = ws / "outputs" / "Q3_xai_validation_preparation_2026-09-26" / "q3_xai_valid120_results.zip"
    add(xai_zip, "artifacts/q3_xai_valid120_results.zip")

    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    handoff_note = (
        "Q3 external audit package. Start with: "
        "阶段4.2_Q3_v1_preflight/external_review_handoff_2026-09-26/README.md. "
        "The ZIP contains historical frozen outputs and read-only audit scripts; "
        "it does not contain the source aligned_50.pkl or MP4 media. "
        "No model training or inference was performed while building this package.\n"
    ).encode("utf-8")
    entries = [{"path": arc, "sha256": file_sha(src), "size": src.stat().st_size} for src, arc in sorted(sources.items(), key=lambda x: x[1])]
    entries.append({"path": "README_EXTERNAL_REVIEW.md", "sha256": hashlib.sha256(handoff_note).hexdigest(), "size": len(handoff_note)})
    manifest_bytes = (json.dumps({"files": entries, "omitted": ["Attachment2 aligned_50.pkl", "Attachment4 source MP4 videos"], "manifest_excludes_itself": True}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in sorted(sources.items(), key=lambda x: x[1]):
            z.write(src, arc)
        z.writestr("README_EXTERNAL_REVIEW.md", handoff_note)
        z.writestr("BUNDLE_MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            output.unlink()
            raise SystemExit("ZIP CRC validation failed")
    print(json.dumps({"status": "BUNDLE_CREATED", "files": len(entries), "zip_bytes": output.stat().st_size, "zip_sha256": file_sha(output), "out": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

