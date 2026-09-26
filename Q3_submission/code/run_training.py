"""Portable reproduction of the frozen Q3 training and architecture-selection protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch

from q3v1.data import EXPECTED_SHA256, TrainScaler, _validate_split, sha256_file
from q3v1.model import Q3Model
from q3v1.train_eval import TrainConfig, collate, evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_candidate(train, valid, scaler, variant, seed, checkpoint, device, config=TrainConfig()):
    set_seed(seed)
    model = Q3Model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_score = float("inf")
    patience_anchor = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        losses = []
        for begin in range(0, len(order), config.batch_size):
            batch = [train[int(i)] for i in order[begin:begin + config.batch_size]]
            text, audio, vision, mask, cls, reg = collate(batch, scaler, device)
            optimizer.zero_grad(set_to_none=True)
            logits, intensity = model(text, audio, vision, mask)
            loss = torch.nn.functional.cross_entropy(logits, cls) + config.regression_weight * torch.mean(torch.abs(intensity - reg))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_metrics = evaluate(model, valid, scaler, device, config.batch_size)
        score = valid_metrics["selection_J"]
        if score < best_score:
            best_score = score
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "validation": valid_metrics,
            }, checkpoint)

        if score < patience_anchor - config.reset_delta:
            patience_anchor = score
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid": valid_metrics})
        if stale_epochs >= config.patience:
            break

    return {"variant": variant, "seed": seed, "best_J": best_score, "epochs_run": len(history), "checkpoint": str(checkpoint), "history": history}


def choose_architecture(runs, tolerance=1e-4):
    means = {variant: float(np.mean([r["best_J"] for r in rows])) for variant, rows in runs.items()}
    if abs(means["B0"] - means["B1"]) <= tolerance:
        return "B0", means
    return min(means, key=means.get), means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-pkl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if sha256_file(args.aligned_pkl) != EXPECTED_SHA256:
        raise ValueError("Attachment2 aligned_50.pkl identity mismatch")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("training output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")

    with args.aligned_pkl.open("rb") as f:
        container = pickle.load(f)
    train = _validate_split(container["train"], "train")
    valid = _validate_split(container["valid"], "valid")
    del container

    if len(train) != 3395 or len(valid) != 728:
        raise ValueError("official train/valid counts differ from frozen protocol")
    if set(s.sample_id for s in train) & set(s.sample_id for s in valid):
        raise ValueError("train/valid sample-id overlap")

    scaler = TrainScaler().fit(train, split="train")
    torch.save({"source_split": "train", "mean": scaler.mean, "std": scaler.std}, args.out / "train_scaler.pt")

    runs = {"B0": [], "B1": []}
    for variant in ("B0", "B1"):
        for seed in (2029, 2030, 2031):
            runs[variant].append(
                fit_candidate(train, valid, scaler, variant, seed, args.out / f"{variant}_seed{seed}.pt", device)
            )

    selected, means = choose_architecture(runs)
    summary = {
        "status": "TRAINING_PROTOCOL_COMPLETE",
        "selected_architecture": selected,
        "architecture_mean_valid_J": means,
        "delivery_seed": 2029,
        "delivery_seed_rule": "fixed by frozen protocol; not selected by valid",
        "selected_checkpoint": f"{selected}_seed2029.pt",
        "runs": runs,
        "test_used_for_selection": False,
        "attachment4_used_for_selection": False,
        "note": "Portable competition reproduction entry; internal server audit gates are intentionally excluded from the submission package."
    }
    (args.out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_architecture": selected, "delivery_seed": 2029, "selected_checkpoint": summary["selected_checkpoint"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
