"""Recompute selected B0 checkpoint metrics on Attachment2 valid, without test selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from q3v1.data import EXPECTED_SHA256, TrainScaler, _validate_split, sha256_file  # noqa: E402
from q3v1.model import Q3Model  # noqa: E402
from q3v1.train_eval import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-pkl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.aligned_pkl) != EXPECTED_SHA256:
        raise ValueError("Attachment2 hash mismatch")
    summary = json.loads((args.run_dir / "training_summary.json").read_text(encoding="utf-8"))
    if (summary["selected_architecture"], summary["selected_checkpoint"]) != ("B0", "B0_seed2029.pt"):
        raise ValueError("selected candidate changed")
    # The PKL container loads all splits, but only train and valid are indexed here.
    with args.aligned_pkl.open("rb") as stream:
        source = pickle.load(stream)
    train = _validate_split(source["train"], "train")
    valid = _validate_split(source["valid"], "valid")
    del source
    if len(train) != 3395 or len(valid) != 728:
        raise ValueError("split counts changed")
    scaler = TrainScaler().fit(train, split="train")
    # This checkpoint contains NumPy arrays, so allow only the required NumPy
    # reconstruction classes while keeping PyTorch's weights-only loader.
    safe_numpy = [_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(safe_numpy):
        saved_scaler = torch.load(args.run_dir / "train_scaler.pt", map_location="cpu", weights_only=True)
    if saved_scaler["source_split"] != "train":
        raise ValueError("saved scaler was not fitted on train")
    scaler_deltas = {key: {
        "mean": float(np.max(np.abs(saved_scaler["mean"][key] - scaler.mean[key]))),
        "std": float(np.max(np.abs(saved_scaler["std"][key] - scaler.std[key]))),
    } for key in ("text", "audio", "vision")}
    checkpoint = torch.load(args.run_dir / summary["selected_checkpoint"], map_location="cpu", weights_only=True)
    model = Q3Model("B0")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    replay = evaluate(model, valid, scaler, torch.device("cpu"))
    reported = checkpoint["validation"]
    deltas = {key: abs(replay[key] - reported[key]) for key in ("accuracy", "macro_f1", "mae", "selection_J")}
    result = {
        "status": "PASS" if (max(deltas.values()) < 0.005 and
                             max(value for row in scaler_deltas.values() for value in row.values()) < 1e-10) else "REVIEW",
        "selected_checkpoint": summary["selected_checkpoint"],
        "checkpoint_epoch": checkpoint["epoch"],
        "train_count": len(train),
        "valid_count": len(valid),
        "test_used_for_selection": False,
        "replay_device": "cpu",
        "replayed_valid_metrics": replay,
        "reported_valid_metrics": reported,
        "absolute_deltas": deltas,
        "comparison_tolerance": 0.005,
        "saved_scaler_vs_refit_max_abs_deltas": scaler_deltas,
        "scaler_comparison_tolerance": 1e-10,
    }
    args.out.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "absolute_deltas": deltas}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
