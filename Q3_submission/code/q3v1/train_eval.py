"""Q3 training/evaluation utilities used by the portable submission scripts."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .data import AlignedSample, TrainScaler
from .model import Q3Model


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


def collate(samples: list[AlignedSample], scaler: TrainScaler, device: torch.device) -> tuple:
    if not samples:
        raise ValueError("empty batch")
    prepared = [scaler.transform(sample) for sample in samples]
    features = [torch.from_numpy(np.stack([item[i] for item in prepared])).to(device) for i in range(3)]
    mask = torch.from_numpy(np.stack([item[3] for item in prepared])).to(device)
    classes = torch.tensor([s.classification_label for s in samples], dtype=torch.long, device=device)
    regression = torch.tensor([s.regression_label for s in samples], dtype=torch.float32, device=device)
    return *features, mask, classes, regression


def metrics(class_true: np.ndarray, class_pred: np.ndarray, reg_true: np.ndarray, reg_pred: np.ndarray) -> dict:
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
    pearson = float(np.corrcoef(reg_true, reg_pred)[0, 1]) if np.std(reg_true) > 0 and np.std(reg_pred) > 0 else None
    macro_f1 = float(np.mean(per_class))
    result = {
        "accuracy": float(np.mean(class_true == class_pred)),
        "macro_f1": macro_f1,
        "per_class_f1": {"Negative": per_class[0], "Neutral": per_class[1], "Positive": per_class[2]},
        "mae": mae,
        "pearson": pearson,
        "selection_J": 0.5 * (1 - macro_f1) + 0.5 * mae / 6.0,
    }
    result["direct_opposite_polarity_rate"] = float(np.mean(
        ((class_pred == 0) & (reg_pred > 0)) | ((class_pred == 2) & (reg_pred < 0))
    ))
    neutral_abs = np.abs(reg_pred[class_pred == 1])
    result["neutral_predicted_abs_regression_mean"] = float(neutral_abs.mean()) if len(neutral_abs) else None
    return result


@torch.no_grad()
def evaluate(model: Q3Model, valid: list[AlignedSample], scaler: TrainScaler, device: torch.device, batch_size: int = 32) -> dict:
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
