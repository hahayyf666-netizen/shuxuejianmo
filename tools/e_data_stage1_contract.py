from __future__ import annotations

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
            "sample_key",
            "text_input",
            "audio",
            "vision",
            "validity_info",
            "label_cls",
            "label_reg",
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
        "forbid_q1_label_driven_transfer": [
            "Q1 labels",
            "Q1 label-based feature selection",
            "Q1 label-based model selection",
            "Q1 label-based hyperparameter choice",
            "Q1 label-based threshold choice",
        ],
        "allow_label_independent_reuse": [
            "data reading code",
            "video decoding code",
            "fixed pretrained tools",
            "label-independent preprocessing functions",
        ],
    },
}


def main():
    required = [
        "attachment1_audit.json",
        "attachment2_audit.json",
        "attachment3_audit.json",
        "attachment4_audit.json",
        "cross_attachment_audit.json",
    ]
    present = {name: (OUT/name).exists() for name in required}
    contract = dict(CONTRACT)
    contract["machine_audit_presence"] = present

    blockers = []
    if not all(present.values()):
        blockers.append("not all attachment/cross audits are present")
    if present["attachment3_audit.json"]:
        a3=json.loads((OUT/"attachment3_audit.json").read_text(encoding="utf-8"))
        if a3.get("versions",{}).get("aligned",{}).get("text_bert_invalid_files"):
            blockers.append("Attachment3 aligned text_bert validation failed")
    if present["cross_attachment_audit.json"]:
        cross=json.loads((OUT/"cross_attachment_audit.json").read_text(encoding="utf-8"))
        if cross.get("status")!="PASS":
            blockers.append("cross-attachment baseline/leakage audit failed")

    contract["stage1_blockers_after_contract"] = blockers
    contract["stage1_status"] = "STOP" if blockers else "READY_FOR_RED_TEAM"

    (OUT/"stage1_data_contract.json").write_text(
        json.dumps(contract,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    lines=[
        "E题 Stage 1 数据与接口规范","="*60,
        f"stage1_status={contract['stage1_status']}",
        f"machine_audit_presence={present}",
        "Q2 aligned=READY; Q2 unaligned=BLOCKED (risk isolation, not final model choice)",
        "masks=validity_mask + synthetic_missing_mask + zero_diagnostic_mask",
        "observed_mask=validity_mask AND NOT synthetic_missing_mask",
        "Q1 label-driven choices -> Q2/Q3: FORBIDDEN",
        "official split: keep; train-internal CV must group by video_id",
        "metrics: Accuracy + Macro-F1 + per-class F1; Weighted-F1 optional",
        f"blockers={blockers}",
    ]
    (OUT/"stage1_data_contract.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
