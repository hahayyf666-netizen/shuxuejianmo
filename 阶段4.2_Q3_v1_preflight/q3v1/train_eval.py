"""Frozen training/evaluation functions. Preflight never calls fit_candidate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch

from .data import AlignedSample, TrainScaler
from .model import Q3Model
from .scope_gate import verify_training_release


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8
    reset_delta: float = 1e-4
    regression_weight: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(samples: list[AlignedSample], scaler: TrainScaler, device: torch.device) -> tuple:
    if not samples:
        raise ValueError("empty batch")
    prepared = [scaler.transform(sample) for sample in samples]
    features = [torch.from_numpy(np.stack([item[i] for item in prepared])).to(device) for i in range(3)]
    mask = torch.from_numpy(np.stack([item[3] for item in prepared])).to(device)
    classes = torch.tensor([s.classification_label for s in samples], dtype=torch.long, device=device)
    regression = torch.tensor([s.regression_label for s in samples], dtype=torch.float32, device=device)
    return *features, mask, classes, regression


def metrics(class_true: np.ndarray, class_pred: np.ndarray,
            reg_true: np.ndarray, reg_pred: np.ndarray) -> dict:
    class_true = np.asarray(class_true, dtype=np.int64)
    class_pred = np.asarray(class_pred, dtype=np.int64)
    reg_true = np.asarray(reg_true, dtype=np.float64)
    reg_pred = np.asarray(reg_pred, dtype=np.float64)
    n = len(class_true)
    if not n or any(len(v) != n for v in (class_pred, reg_true, reg_pred)):
        raise ValueError("metric input length mismatch")
    if not np.isin(class_true, [0, 1, 2]).all() or not np.isin(class_pred, [0, 1, 2]).all():
        raise ValueError("class encoding must be 0/1/2")
    if not np.isfinite(reg_true).all() or not np.isfinite(reg_pred).all():
        raise ValueError("nonfinite regression prediction")
    per_class = []
    for label in (0, 1, 2):
        tp = np.sum((class_true == label) & (class_pred == label))
        fp = np.sum((class_true != label) & (class_pred == label))
        fn = np.sum((class_true == label) & (class_pred != label))
        per_class.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    mae = float(np.mean(np.abs(reg_true - reg_pred)))
    pearson = (float(np.corrcoef(reg_true, reg_pred)[0, 1])
               if np.std(reg_true) > 0 and np.std(reg_pred) > 0 else None)
    macro_f1 = float(np.mean(per_class))
    result = {"accuracy": float(np.mean(class_true == class_pred)), "macro_f1": macro_f1,
              "per_class_f1": {"Negative": per_class[0], "Neutral": per_class[1], "Positive": per_class[2]},
              "mae": mae, "pearson": pearson,
              "selection_J": 0.5 * (1 - macro_f1) + 0.5 * mae / 6.0}
    # Direct polarity conflict excludes Neutral; no invented Neutral regression threshold.
    result["direct_opposite_polarity_rate"] = float(np.mean(
        ((class_pred == 0) & (reg_pred > 0)) | ((class_pred == 2) & (reg_pred < 0))))
    neutral_abs = np.abs(reg_pred[class_pred == 1])
    result["neutral_predicted_abs_regression_mean"] = float(neutral_abs.mean()) if len(neutral_abs) else None
    return result


@torch.no_grad()
def evaluate(model: Q3Model, valid: list[AlignedSample], scaler: TrainScaler,
             device: torch.device, batch_size: int = 32) -> dict:
    model.eval()
    actual_cls, pred_cls, actual_reg, pred_reg = [], [], [], []
    for start in range(0, len(valid), batch_size):
        batch = valid[start:start + batch_size]
        text, audio, vision, mask, cls, reg = collate(batch, scaler, device)
        logits, intensity = model(text, audio, vision, mask)
        actual_cls.extend(cls.cpu().numpy().tolist())
        pred_cls.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        actual_reg.extend(reg.cpu().numpy().tolist())
        pred_reg.extend(intensity.cpu().numpy().tolist())
    return metrics(np.array(actual_cls), np.array(pred_cls), np.array(actual_reg), np.array(pred_reg))


def fit_candidate(*, train: list[AlignedSample], valid: list[AlignedSample], scaler: TrainScaler,
                  variant: str, seed: int, checkpoint: Path, device: torch.device,
                  config: TrainConfig = TrainConfig(), mapping_gate_report: Path | None = None) -> dict:
    """Formal training entry; a passing server preflight and final scope are required."""
    if mapping_gate_report is None or not mapping_gate_report.is_file():
        raise PermissionError("Q3 server preflight gate missing; formal training forbidden")
    verify_training_release(mapping_gate_report)
    if not train or not valid or scaler.source_split != "train":
        raise ValueError("train/valid and train scaler required")
    if set(s.sample_id for s in train) & set(s.sample_id for s in valid):
        raise ValueError("train/valid ID overlap")
    if seed not in (2029, 2030, 2031):
        raise ValueError("seed outside frozen comparison plan")
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
            loss = torch.nn.functional.cross_entropy(logits, cls) + \
                   config.regression_weight * torch.mean(torch.abs(intensity - reg))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_metrics = evaluate(model, valid, scaler, device, config.batch_size)
        score = valid_metrics["selection_J"]
        if score < best_score:
            best_score = score
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"variant": variant, "seed": seed, "epoch": epoch,
                        "state_dict": model.state_dict(), "validation": valid_metrics}, checkpoint)
        if score < patience_anchor - config.reset_delta:
            patience_anchor = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid": valid_metrics})
        if stale_epochs >= config.patience:
            break
    return {"variant": variant, "seed": seed, "best_J": best_score,
            "epochs_run": len(history), "history": history,
            "checkpoint": str(checkpoint)}


def choose_architecture(runs: dict[str, list[dict]], tolerance: float = 1e-4) -> str:
    if set(runs) != {"B0", "B1"} or any(len(runs[v]) != 3 for v in runs):
        raise ValueError("exactly three B0 and three B1 seeds required")
    expected = {2029, 2030, 2031}
    if any({r["seed"] for r in runs[v]} != expected for v in runs):
        raise ValueError("frozen seeds missing")
    means = {v: float(np.mean([r["best_J"] for r in runs[v]])) for v in runs}
    if abs(means["B0"] - means["B1"]) <= tolerance:
        return "B0"  # strictly fewer parameters
    return min(means, key=means.get)
