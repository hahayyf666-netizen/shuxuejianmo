from __future__ import annotations

# Stage 1 contract: machine gate is intentionally fail-closed.

import json
from pathlib import Path

OUT = Path("_e_data_audit_out")
OUT.mkdir(exist_ok=True)


CONTRACT = {
    "scope": "Stage 1 public data/interface contract; does not choose the final Q2/Q3 model.",
    "official_split_policy": {
        "train": "parameter learning",
        "valid": "model structure / hyperparameter / decision-threshold selection",
        "test": "never used for selection; optional one-shot holdout after model freeze",
        "resplitting": "keep official splits; if train-internal CV is needed, group by video_id, never random clip-level CV",
    },
    "classification_metrics": {
        "primary": ["Accuracy", "Macro-F1", "per-class F1"],
        "optional": ["Weighted-F1"],
        "rule": "metric variant must be fixed before viewing model-comparison results",
    },
    "q2_model_facing_interface": {
        "common_fields": [
            "sample_key", "text_input", "audio", "vision",
            "validity_info", "label_cls", "label_reg"
        ],
        "aligned": {
            "status": "READY",
            "allowed_dependencies": ["text_bert", "audio", "vision"],
            "forbidden_dependencies": ["text", "raw_text", "id"],
            "adapter": [
                "assert text_bert is integer-valued",
                "assert attention mask is binary",
                "cast text_bert to int64",
                "cast audio and vision to float32",
                "derive sample_key from source metadata/file identity rather than model input",
            ],
        },
        "unaligned": {
            "status": "BLOCKED",
            "allowed_if_reopened": ["raw_text", "audio", "vision"],
            "forbidden_dependencies": ["text", "text_bert", "id", "audio_lengths", "vision_lengths"],
            "block_reasons": [
                "Attachment3 unaligned lacks audio_lengths and vision_lengths",
                "Attachment2/4 vision_lengths semantics are not reliably closed",
            ],
        },
        "hard_rule": "No Q2 model may depend on a field unavailable at Attachment3 final inference. The same-input-interface requirement is evaluated at the normalized model-facing interface, not raw PKL key equality.",
    },
    "q3_model_facing_interface": {
        "aligned": {
            "status": "READY",
            "attachment2_available": ["raw_text", "text", "text_bert", "audio", "vision", "id", "labels"],
            "attachment4_available": ["raw_text", "text", "text_bert", "audio", "vision", "id", "original_video"],
            "predictive_modalities": ["text_representation", "audio", "vision"],
            "text_representation": {
                "one_of": ["text", "text_bert"],
                "exactly_one": True,
                "rule": "text and text_bert are alternative representations of the same text modality and must not be used simultaneously as separate modalities"
            },
            "raw_text_role": "evidence/back-tracing and display; not an additional parallel predictive modality under the frozen Stage 1 interface",
            "forbidden_predictive_dependencies": ["id", "original_video", "simultaneous text and text_bert"],
            "original_video_role": "evidence back-tracing / explanation display only; Attachment2 training has no raw video, so it is not a predictive model input",
        },
        "unaligned": {
            "status": "BLOCKED",
            "attachment2_available": ["raw_text", "text", "text_bert", "audio", "vision", "audio_lengths", "vision_lengths", "id", "labels"],
            "attachment4_available": ["raw_text", "text", "text_bert", "audio", "vision", "audio_lengths", "vision_lengths", "id", "original_video"],
            "predictive_modalities_if_reopened": ["text_representation", "audio", "vision"],
            "text_representation_if_reopened": {
                "one_of": ["text", "text_bert"],
                "exactly_one": True
            },
            "block_reasons": [
                "vision_lengths semantics are not reliably closed in Attachment2/4",
                "unaligned validity handling must be reopened and validated before model admission",
            ],
        },
        "hard_rule": "Q3 predictive inputs must be available in both Attachment2 training/validation and Attachment4 final inference. Raw Attachment4 video is reserved for evidence back-tracing, not predictive training.",
    },
    "field_availability_matrix": {
        "Q2_aligned_common_predictive": ["text_bert", "audio", "vision"],
        "Q2_unaligned_common_predictive_if_reopened": ["raw_text", "audio", "vision"],
        "Q3_aligned_common_available": ["raw_text", "text", "text_bert", "audio", "vision"],
        "Q3_aligned_predictive_contract": ["one_of(text,text_bert)", "audio", "vision"],
        "Q3_unaligned_common_available_if_reopened": ["raw_text", "text", "text_bert", "audio", "vision", "audio_lengths", "vision_lengths"],
        "Q3_unaligned_predictive_contract_if_reopened": ["one_of(text,text_bert)", "audio", "vision"],
    },
    "mask_contract": {
        "validity_mask": "padding/non-valid sequence-position mask; for aligned text_bert routes use the frozen attention-mask semantics, not feature-zero tests",
        "synthetic_missing_mask": "only positions deliberately masked by our training/robustness experiment; generated and stored explicitly",
        "zero_diagnostic_mask": "records whether an input feature row is all-zero for diagnostics only; never treated as missing ground truth",
        "observed_mask": "validity_mask AND NOT synthetic_missing_mask",
        "final_attachment3_rule": "Do not assume a ground-truth artificial-missing mask exists. A model may use a missing indicator only if it is deterministically derivable from Attachment3 inputs by a pre-frozen, prediction-independent rule; otherwise it must not depend on such a mask.",
        "forbidden": [
            "missing_mask = (feature == 0) as ground truth",
            "using model outputs to choose or repair masks",
        ],
    },
    "risk_isolation": {
        "aligned": "READY for later model comparison/selection; this is not a final choice of aligned",
        "unaligned": "BLOCKED until its STOP is explicitly reopened and resolved",
        "unaligned_forbidden_repairs": [
            "vision_length = last_nonzero",
            "blind truncation to vision_lengths",
            "choosing a repair rule after inspecting prediction performance",
        ],
    },
    "q1_q2q3_leakage_firewall": {
        "forbid_any_q1_data_fitted_state_transfer": [
            "Q1 labels",
            "Q1 label-based feature/model/hyperparameter/threshold selection",
            "normalization statistics estimated from Q1 samples",
            "PCA/dimensionality-reduction parameters fitted on Q1 samples",
            "clustering/codebook state fitted on Q1 samples",
            "self-supervised adaptation performed on Q1 videos",
            "feature extractors fine-tuned or adapted on Q1 samples",
            "any preprocessing parameter/statistic/model state fitted, estimated, calibrated, adapted, or selected using Q1 samples",
        ],
        "allow_only_q1_data_independent_reuse": [
            "stateless data reading code",
            "stateless video decoding code",
            "fixed pretrained tools with no Q1 adaptation",
            "preprocessing functions whose parameters are fixed independently of Q1 data",
        ],
        "reason": "Attachment1 overlaps Attachment2 train/test; data-derived Q1 state can leak Attachment2 test information even without labels",
    },
}


def attachment2_ok(a2):
    try:
        video_ov = a2.get("label_xlsx", {}).get("video_id_split_overlaps", {})
        if not video_ov or any(v != 0 for v in video_ov.values()):
            return False, f"official split video_id overlap is nonzero or unavailable: {video_ov}"
        for fname in ["aligned_50.pkl", "unaligned_50.pkl"]:
            rep = a2[fname]
            if any(v != 0 for v in rep.get("split_id_overlaps", {}).values()):
                return False, f"{fname} split id overlap is nonzero"
            expected = {"train": 3395, "valid": 728, "test": 727}
            for split, n in expected.items():
                sp = rep["splits"][split]
                if sp.get("n") != n:
                    return False, f"{fname}/{split} count mismatch"
                for field in ["text", "text_bert", "audio", "vision"]:
                    meta = sp.get(field, {})
                    if meta.get("nan", 0) or meta.get("inf", 0):
                        return False, f"{fname}/{split}/{field} contains NaN/Inf"
        return True, None
    except Exception as e:
        return False, f"attachment2 audit parse/check error: {e!r}"


def main():
    required = [
        "attachment1_audit.json",
        "attachment2_audit.json",
        "attachment3_audit.json",
        "attachment4_audit.json",
        "cross_attachment_audit.json",
    ]
    present = {name: (OUT / name).exists() for name in required}
    contract = dict(CONTRACT)
    contract["machine_audit_presence"] = present

    blockers = []
    statuses = {}

    if not all(present.values()):
        blockers.append("not all attachment/cross audits are present")

    if present["attachment1_audit.json"]:
        a1 = json.loads((OUT / "attachment1_audit.json").read_text(encoding="utf-8"))
        statuses["attachment1"] = a1.get("status")
        if a1.get("status") != "PASS":
            blockers.append("Attachment1 machine audit is not PASS")

    if present["attachment2_audit.json"]:
        a2 = json.loads((OUT / "attachment2_audit.json").read_text(encoding="utf-8"))
        ok, why = attachment2_ok(a2)
        statuses["attachment2"] = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append(why)

    if present["attachment3_audit.json"]:
        a3 = json.loads((OUT / "attachment3_audit.json").read_text(encoding="utf-8"))
        statuses["attachment3"] = a3.get("status")
        if not str(a3.get("status", "")).startswith("PASS"):
            blockers.append("Attachment3 machine audit is not PASS")
        if a3.get("versions", {}).get("aligned", {}).get("text_bert_invalid_files"):
            blockers.append("Attachment3 aligned text_bert validation failed")

    if present["attachment4_audit.json"]:
        a4 = json.loads((OUT / "attachment4_audit.json").read_text(encoding="utf-8"))
        statuses["attachment4"] = a4.get("status")
        if not str(a4.get("status", "")).startswith("PASS"):
            blockers.append("Attachment4 machine audit is not PASS")

    if present["cross_attachment_audit.json"]:
        cross = json.loads((OUT / "cross_attachment_audit.json").read_text(encoding="utf-8"))
        statuses["cross_attachment"] = cross.get("status")
        if cross.get("status") != "PASS":
            blockers.append("cross-attachment baseline/leakage audit failed")

    contract["machine_audit_statuses"] = statuses
    contract["stage1_blockers_after_contract"] = blockers
    contract["stage1_status"] = "STOP" if blockers else "READY_FOR_RED_TEAM"

    (OUT / "stage1_data_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "E题 Stage 1 数据与接口规范",
        "=" * 60,
        f"stage1_status={contract['stage1_status']}",
        f"machine_audit_presence={present}",
        f"machine_audit_statuses={statuses}",
        "Q2 aligned=READY; Q2 unaligned=BLOCKED (risk isolation, not final model choice)",
        "Q3 aligned=READY; Q3 text representation=exactly one of(text,text_bert); Q3 unaligned=BLOCKED; Attachment4 raw video is explanation/back-tracing only",
        "masks=validity_mask + synthetic_missing_mask + zero_diagnostic_mask",
        "observed_mask=validity_mask AND NOT synthetic_missing_mask",
        "Any Q1-data-fitted/estimated/calibrated/adapted/selected state -> Q2/Q3: FORBIDDEN",
        "official split: keep; train-internal CV must group by video_id",
        "metrics: Accuracy + Macro-F1 + per-class F1; Weighted-F1 optional",
        f"blockers={blockers}",
    ]
    (OUT / "stage1_data_contract.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
