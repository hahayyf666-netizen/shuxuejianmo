"""Read-only audit of the six frozen Q3 training runs and checkpoints."""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import torch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from q3v1.model import Q3Model  # noqa: E402

PREFIX = "b0b1_v1_cuda128/"
SEEDS = (2029, 2030, 2031)
EXPECTED_NAMES = {
    *(f"{PREFIX}{variant}_seed{seed}.pt" for variant in ("B0", "B1") for seed in SEEDS),
    f"{PREFIX}training_summary.json",
    f"{PREFIX}training_progress.json",
    f"{PREFIX}train_scaler.pt",
    "b0b1_v1_cuda128.stdout.log",
    "b0b1_v1_cuda128.stderr.log",
}


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def audit(archive: Path) -> dict:
    checks: dict[str, bool] = {}
    with ZipFile(archive) as bundle:
        checks["zip_crc"] = bundle.testzip() is None
        names = set(bundle.namelist())
        checks["exact_file_set"] = names == EXPECTED_NAMES and len(bundle.namelist()) == len(names)
        files = {name: bundle.read(name) for name in names}
    if not all(checks.values()):
        return {"status": "FAIL", "checks": checks, "archive_sha256": digest(archive.read_bytes())}

    summary = json.loads(files[f"{PREFIX}training_summary.json"])
    progress = json.loads(files[f"{PREFIX}training_progress.json"])
    stdout = json.loads(files["b0b1_v1_cuda128.stdout.log"].decode("utf-8").strip())
    gate_path = STAGE / "server_preflight_drx_cuda128_2026-09-26" / "q3_server_preflight_gate.json"
    scope_path = STAGE / "stage_d_explanation_scope_finalization" / "explanation_scope_contract.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    checks["summary_complete"] = summary["status"] == "B0_B1_VALID_SELECTION_COMPLETE"
    checks["preflight_gate_identity"] = (summary["gate_sha256"] == digest(gate_path.read_bytes())
                                          and gate["status"] == "PASS")
    checks["scope_identity"] = summary["scope_contract_sha256"] == digest(scope_path.read_bytes())
    checks["aligned_input_identity"] = summary["aligned_pkl_sha256"] == (
        "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd")
    checks["scaler_identity"] = summary["scaler_sha256"] == digest(files[f"{PREFIX}train_scaler.pt"])
    checks["device"] = summary["device"] == "cuda:1"
    checks["logs"] = (not files["b0b1_v1_cuda128.stderr.log"]
                      and stdout["status"] == summary["status"]
                      and stdout["selected_architecture"] == summary["selected_architecture"]
                      and stdout["test_used_for_selection"] is False)
    checks["split_flags"] = (summary["test_used_for_selection"] is False
                             and summary["attachment4_used_for_selection"] is False
                             and progress["test_used_for_selection"] is False
                             and progress["attachment4_used_for_selection"] is False)
    checks["progress_snapshot"] = (progress["status"] == "IN_PROGRESS"
                                    and progress["completed_runs"] == summary["runs"])
    checks["candidate_set"] = (set(summary["runs"]) == {"B0", "B1"} and all(
        [run["seed"] for run in summary["runs"][variant]] == list(SEEDS)
        and all(run["variant"] == variant for run in summary["runs"][variant])
        for variant in ("B0", "B1")))

    run_rows = []
    if checks["candidate_set"]:
        for variant in ("B0", "B1"):
            for run in summary["runs"][variant]:
                seed = run["seed"]
                history = run["history"]
                scores = [h["valid"]["selection_J"] for h in history]
                expected_epoch = scores.index(min(scores)) + 1
                name = f"{PREFIX}{variant}_seed{seed}.pt"
                checkpoint = torch.load(BytesIO(files[name]), map_location="cpu", weights_only=True)
                model = Q3Model(variant)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                finite_weights = all(bool(torch.isfinite(t).all()) for t in checkpoint["state_dict"].values())
                row = {
                    "variant": variant,
                    "seed": seed,
                    "epochs_run": run["epochs_run"],
                    "best_epoch": checkpoint["epoch"],
                    "best_J": run["best_J"],
                    "checkpoint_sha256": digest(files[name]),
                    "checkpoint_size": len(files[name]),
                    "parameter_count": model.parameter_count,
                    "history_valid": (
                        len(history) == run["epochs_run"]
                        and 1 <= len(history) <= 60
                        and [h["epoch"] for h in history] == list(range(1, len(history) + 1))
                        and all(math.isfinite(h["train_loss"]) for h in history)
                        and all(math.isfinite(s) for s in scores)
                        and math.isclose(run["best_J"], min(scores), abs_tol=1e-12)
                    ),
                    "checkpoint_valid": (
                        checkpoint["variant"] == variant
                        and checkpoint["seed"] == seed
                        and checkpoint["epoch"] == expected_epoch
                        and checkpoint["validation"] == history[expected_epoch - 1]["valid"]
                        and finite_weights
                        and model.parameter_count == (68932 if variant == "B0" else 125380)
                    ),
                }
                run_rows.append(row)
    checks["run_histories"] = len(run_rows) == 6 and all(r["history_valid"] for r in run_rows)
    checks["checkpoints"] = len(run_rows) == 6 and all(r["checkpoint_valid"] for r in run_rows)

    means = {v: sum(r["best_J"] for r in summary["runs"][v]) / 3 for v in ("B0", "B1")}
    expected_selection = "B0" if abs(means["B0"] - means["B1"]) <= 1e-4 else min(means, key=means.get)
    checks["selection"] = (summary["selected_architecture"] == expected_selection
                            and summary["delivery_seed"] == 2029
                            and summary["selected_checkpoint"] == f"{expected_selection}_seed2029.pt")
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "archive_sha256": digest(archive.read_bytes()),
        "archive_size": archive.stat().st_size,
        "file_sha256": {name: digest(files[name]) for name in sorted(files)},
        "mean_best_J": means,
        "selected_architecture": summary["selected_architecture"],
        "selected_checkpoint": summary["selected_checkpoint"],
        "runs": run_rows,
        "progress_status_note": "IN_PROGRESS is the last interim snapshot; final status is in training_summary.json",
        "checkpoint_content_published": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
