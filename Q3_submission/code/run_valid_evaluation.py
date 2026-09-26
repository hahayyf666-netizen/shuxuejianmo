"""Reproduce frozen B0 validation metrics on Attachment2 aligned_50."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct

from q3v1.data import TrainScaler, load_split
from q3v1.model import Q3Model
from q3v1.train_eval import collate, metrics


def load_scaler(path: Path) -> TrainScaler:
    with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("source_split") != "train":
        raise ValueError("scaler source is not train")
    scaler = TrainScaler()
    scaler.mean = {k: np.asarray(v, dtype=np.float64) for k, v in saved["mean"].items()}
    scaler.std = {k: np.asarray(v, dtype=np.float64) for k, v in saved["std"].items()}
    scaler.source_split = "train"
    return scaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-pkl", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scaler", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    valid = load_split(args.aligned_pkl, "valid", verify_hash=True)
    if len(valid) != 728:
        raise ValueError(f"expected 728 valid samples, got {len(valid)}")

    scaler = load_scaler(args.scaler)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "B0" or checkpoint.get("seed") != 2029:
        raise ValueError("checkpoint is not frozen B0_seed2029")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")

    model = Q3Model("B0").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    true_cls, pred_cls, true_reg, pred_reg = [], [], [], []
    with torch.inference_mode():
        for begin in range(0, len(valid), 32):
            batch = valid[begin:begin + 32]
            text, audio, vision, mask, cls, reg = collate(batch, scaler, device)
            logits, intensity = model(text, audio, vision, mask)
            true_cls.extend(cls.cpu().numpy().tolist())
            pred_cls.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            true_reg.extend(reg.cpu().numpy().tolist())
            pred_reg.extend(intensity.cpu().numpy().tolist())

    result = metrics(np.asarray(true_cls), np.asarray(pred_cls), np.asarray(true_reg), np.asarray(pred_reg))
    errors = np.asarray(pred_reg, dtype=np.float64) - np.asarray(true_reg, dtype=np.float64)
    result["rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    payload = {"status": "VALID_EVALUATION_COMPLETE", "sample_count": len(valid), "checkpoint_variant": checkpoint.get("variant"), "checkpoint_seed": checkpoint.get("seed"), "metrics": result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
