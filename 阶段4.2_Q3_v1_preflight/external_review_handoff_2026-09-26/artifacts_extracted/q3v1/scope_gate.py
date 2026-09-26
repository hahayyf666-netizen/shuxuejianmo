"""Validate the C-4 explanation scope and a later server training release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_GATES = {
    "text": "TEXT_MAPPING_PASS_ATTACHMENT4_SCOPE",
    "audio": "AUDIO_MAPPING_BLOCKED",
    "vision": "VISION_MAPPING_BLOCKED",
    "explanation": "FEATURE_SPACE_ATTRIBUTION_ALLOWED",
}
EXPECTED_MAPPING = {"text": "verified_text", "audio": "index_only", "vision": "index_only"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_child(root: Path, relative: str, expected_hash: str) -> Path:
    if not isinstance(relative, str) or not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise PermissionError("missing scope evidence path/hash")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file() or sha256(target) != expected_hash:
        raise PermissionError("scope evidence missing, outside bundle, or SHA-256 mismatch")
    return target


def verify_scope_contract(contract_path: Path) -> dict:
    root = contract_path.resolve().parent.parent
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (contract.get("scope_version") != "q3-modality-dependent-evidence-scope-v1" or
            contract.get("feature_version") != "aligned_50" or
            contract.get("predictor_text_representation") != "text" or
            contract.get("gates") != EXPECTED_GATES or
            contract.get("mapping_status_by_modality") != EXPECTED_MAPPING or
            contract.get("formal_training_release") != "PENDING_SERVER_PREFLIGHT" or
            contract.get("allowed_methods") != ["exact_3_player_shapley", "conditional_integrated_gradients"] or
            contract.get("attachment4_verified_text_sample_ids") != [f"{i:02d}" for i in range(1, 21)]):
        raise PermissionError("explanation scope contract differs from reviewed C-4 boundary")
    required_prohibitions = {"audio_seconds", "vision_frames", "unverified_channel_physical_semantics",
                             "causal_emotion_contribution", "explanation_accuracy"}
    if not required_prohibitions.issubset(set(contract.get("prohibited_claims", []))):
        raise PermissionError("required interpretation prohibitions missing")
    c4_path = _checked_child(root, contract["c4_gate_relative_path"], contract["c4_gate_sha256"])
    _checked_child(root, contract["c4_text_trace_relative_path"], contract["c4_text_trace_sha256"])
    preserved = contract.get("preserved_artifact_sha256", {})
    if set(preserved) != {"q3v1/model.py", "q3v1/explain.py", "frozen_config.json"}:
        raise PermissionError("original model/explanation artifact identities missing")
    for relative, expected in preserved.items():
        _checked_child(root, relative, expected)
    c4 = json.loads(c4_path.read_text(encoding="utf-8"))
    if (c4["text"]["strict_recovered_content_rows"] != 564 or
            c4["text"]["official_content_rows"] != 564 or
            c4["text"]["all_50_positions_reconstructed"] != 1000 or
            c4["text"]["maximum_absolute_error"] > 1e-4 or
            c4["audio"]["strict_recovered_74D_rows"] != 0 or
            c4["audio"]["official_unaligned_rows"] != 3677 or
            c4["vision"]["strict_recovered_35D_rows"] != 0 or
            c4["vision"]["official_unaligned_rows"] != 2593 or
            c4["text"]["mapping_status_for_tested_samples"] != "verified_text" or
            c4["audio"]["mapping_status"] != "index_only" or
            c4["vision"]["mapping_status"] != "index_only"):
        raise PermissionError("C-4 machine result disagrees with explanation scope")
    return contract


def verify_training_release(gate_path: Path) -> dict:
    """Never release training on the explanation scope alone; require server proof."""
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("gate_version") != "q3-server-preflight-v1" or gate.get("status") != "PASS":
        raise PermissionError("server preflight training gate is not PASS")
    if gate.get("gates") != {**EXPECTED_GATES, "data": "PASS", "interface": "PASS", "xai": "PASS", "environment": "PASS"}:
        raise PermissionError("server preflight gates incomplete")
    root = Path(__file__).resolve().parent.parent
    if not gate_path.resolve().is_relative_to(root):
        raise PermissionError("training gate must reside within the Q3 package")
    contract_path = _checked_child(root, gate.get("scope_contract_path"), gate.get("scope_contract_sha256"))
    verify_scope_contract(contract_path)
    machine_path = _checked_child(root, gate.get("machine_report_path"), gate.get("machine_report_sha256"))
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    environment_path = _checked_child(root, gate.get("environment_report_path"),
                                      gate.get("environment_report_sha256"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if (not environment.get("hostname") or not environment.get("python_executable") or
            not environment.get("torch") or not isinstance(environment.get("cuda_available"), bool) or
            machine.get("environment", {}).get("hostname") != environment["hostname"]):
        raise PermissionError("server environment report incomplete or inconsistent")
    if (machine.get("aligned_pkl_sha256") != "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd" or
            machine.get("status") != "preflight_train_valid_smoke_pass_scope_gate_external" or
            machine.get("split_counts") != {"train": 3395, "valid": 728, "test_accessed_for_selection": False} or
            machine.get("train_valid_id_overlap") != 0 or
            any(item.get("optimizer_steps") != 0 or not item.get("loss_finite") or
                not item.get("backward_gradients_present") for item in machine.get("models", {}).values()) or
            set(machine.get("models", {})) != {"B0", "B1"} or
            machine.get("attribution_numeric_smoke", {}).get("ig_numerical_status") != "pass"):
        raise PermissionError("server machine report does not satisfy train/valid and numeric gates")
    if gate.get("unit_tests_exit_code") != 0 or gate.get("test_used_for_model_selection") is not False:
        raise PermissionError("unit tests or test isolation failed")
    for stem in ("unit_tests", "data_smoke"):
        exit_file = _checked_child(root, gate.get(stem + "_exit_file_path"),
                                   gate.get(stem + "_exit_file_sha256"))
        if exit_file.read_text(encoding="utf-8").strip() != "0":
            raise PermissionError(f"{stem} process did not exit successfully")
    return gate
