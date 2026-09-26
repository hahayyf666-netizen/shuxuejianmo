"""Train frozen B0/B1 on Attachment2 train; select architecture using valid only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import torch

from q3v1.data import EXPECTED_SHA256, TrainScaler, _validate_split, sha256_file
from q3v1.scope_gate import sha256, verify_training_release
from q3v1.train_eval import choose_architecture, fit_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-pkl", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    gate = verify_training_release(args.gate)
    if sha256_file(args.aligned_pkl) != EXPECTED_SHA256:
        raise PermissionError("Attachment2 aligned file SHA-256 mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device unavailable")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("formal training output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    with args.aligned_pkl.open("rb") as stream:
        container = pickle.load(stream)
    train = _validate_split(container["train"], "train")
    valid = _validate_split(container["valid"], "valid")
    del container
    if len(train) != 3395 or len(valid) != 728 or set(s.sample_id for s in train) & set(s.sample_id for s in valid):
        raise ValueError("official train/valid split count or isolation mismatch")
    scaler = TrainScaler().fit(train, split="train")
    scaler_path = out / "train_scaler.pt"
    torch.save({"source_split": "train", "mean": scaler.mean, "std": scaler.std}, scaler_path)
    runs = {"B0": [], "B1": []}
    for variant in ("B0", "B1"):
        for seed in (2029, 2030, 2031):
            checkpoint = out / f"{variant}_seed{seed}.pt"
            run = fit_candidate(train=train, valid=valid, scaler=scaler, variant=variant,
                                seed=seed, checkpoint=checkpoint, device=device,
                                mapping_gate_report=args.gate)
            runs[variant].append(run)
            (out / "training_progress.json").write_text(json.dumps({
                "status": "IN_PROGRESS", "completed_runs": runs,
                "test_used_for_selection": False, "attachment4_used_for_selection": False,
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selected = choose_architecture(runs)
    summary = {
        "status": "B0_B1_VALID_SELECTION_COMPLETE",
        "selected_architecture": selected,
        "delivery_seed": 2029,
        "selected_checkpoint": f"{selected}_seed2029.pt",
        "runs": runs,
        "gate_sha256": sha256(args.gate),
        "scope_contract_sha256": gate["scope_contract_sha256"],
        "aligned_pkl_sha256": EXPECTED_SHA256,
        "scaler_sha256": sha256(scaler_path),
        "device": str(device),
        "python_executable": sys.executable,
        "test_used_for_selection": False,
        "attachment4_used_for_selection": False,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "selected_architecture": selected,
                      "selected_checkpoint": str(out / summary["selected_checkpoint"]),
                      "test_used_for_selection": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
