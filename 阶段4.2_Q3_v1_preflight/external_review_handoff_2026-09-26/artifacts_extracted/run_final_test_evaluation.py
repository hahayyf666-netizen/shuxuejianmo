"""One-shot evaluation of the already frozen B0 checkpoint on Attachment2 test.

This script is intentionally separate from training and Attachment4 inference:
the test split is read only after B0_seed2029 and the train-fitted scaler are
fixed.  It never chooses an architecture, changes a threshold, or writes back
to the Attachment4 results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from q3v1.data import EXPECTED_SHA256, TrainScaler, _validate_split, sha256_file  # noqa: E402
from q3v1.model import Q3Model  # noqa: E402
from q3v1.train_eval import metrics  # noqa: E402


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_scaler(path: Path) -> TrainScaler:
    with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("source_split") != "train":
        raise ValueError("saved scaler is not train-fitted")
    scaler = TrainScaler()
    scaler.mean = {key: np.asarray(value, dtype=np.float64) for key, value in saved["mean"].items()}
    scaler.std = {key: np.asarray(value, dtype=np.float64) for key, value in saved["std"].items()}
    scaler.source_split = "train"
    return scaler


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-pkl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    aligned = args.aligned_pkl.resolve()
    checkpoint_path = args.checkpoint.resolve()
    scaler_path = args.scaler.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"test output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if sha256_file(aligned) != EXPECTED_SHA256:
        raise ValueError("aligned PKL SHA-256 mismatch")

    with aligned.open("rb") as fh:
        container = pickle.load(fh)
    required = {"train", "valid", "test"}
    if set(container) < required:
        raise ValueError(f"missing split keys: {sorted(required - set(container))}")
    train_ids = [str(x) for x in container["train"]["id"]]
    valid_ids = [str(x) for x in container["valid"]["id"]]
    test = _validate_split(container["test"], "test")
    test_ids = [sample.sample_id for sample in test]
    if len(train_ids) != 3395 or len(valid_ids) != 728:
        raise ValueError("train/valid counts differ from the frozen training contract")
    if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
        raise ValueError("split ID overlap detected")
    del container

    scaler = load_train_scaler(scaler_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "B0" or checkpoint.get("seed") != 2029:
        raise ValueError("checkpoint is not the frozen B0_seed2029 delivery checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    model = Q3Model("B0").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    true_cls: list[int] = []
    pred_cls: list[int] = []
    true_reg: list[float] = []
    pred_reg: list[float] = []
    rows: list[dict] = []
    batch_size = 32
    with torch.no_grad():
        for begin in range(0, len(test), batch_size):
            batch = test[begin:begin + batch_size]
            prepared = [scaler.transform(sample) for sample in batch]
            text = torch.from_numpy(np.stack([item[0] for item in prepared])).to(device)
            audio = torch.from_numpy(np.stack([item[1] for item in prepared])).to(device)
            vision = torch.from_numpy(np.stack([item[2] for item in prepared])).to(device)
            mask = torch.from_numpy(np.stack([item[3] for item in prepared])).to(device)
            logits, intensity = model(text, audio, vision, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = logits.argmax(dim=-1).cpu().numpy()
            intensity_np = intensity.cpu().numpy()
            for sample, probability, class_index, reg_value in zip(batch, probs, pred, intensity_np):
                true_cls.append(sample.classification_label)
                pred_cls.append(int(class_index))
                true_reg.append(float(sample.regression_label))
                pred_reg.append(float(reg_value))
                rows.append({
                    "sample_id": sample.sample_id,
                    "true_class": CLASS_NAMES[sample.classification_label],
                    "predicted_class": CLASS_NAMES[int(class_index)],
                    "p_negative": float(probability[0]),
                    "p_neutral": float(probability[1]),
                    "p_positive": float(probability[2]),
                    "true_intensity": float(sample.regression_label),
                    "predicted_intensity": float(reg_value),
                })

    base_metrics = metrics(np.asarray(true_cls), np.asarray(pred_cls), np.asarray(true_reg), np.asarray(pred_reg))
    errors = np.asarray(pred_reg, dtype=np.float64) - np.asarray(true_reg, dtype=np.float64)
    base_metrics["rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    base_metrics["max_abs_error"] = float(np.max(np.abs(errors)))
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for truth, prediction in zip(true_cls, pred_cls):
        confusion[truth][prediction] += 1

    summary = {
        "status": "FINAL_TEST_EVALUATION_COMPLETE",
        "evaluated_split": "test",
        "sample_count": len(test),
        "train_count_checked": len(train_ids),
        "valid_count_checked": len(valid_ids),
        "metrics": base_metrics,
        "confusion_matrix": {"class_order": list(CLASS_NAMES), "rows_true_cols_pred": confusion},
        "selected_architecture": "B0",
        "selected_checkpoint": "B0_seed2029.pt",
        "checkpoint_sha256": digest(checkpoint_path),
        "scaler_sha256": digest(scaler_path),
        "aligned_pkl_sha256": digest(aligned),
        "test_evaluated": True,
        "test_used_for_selection": False,
        "model_or_threshold_changed": False,
        "attachment4_results_modified": False,
        "labels_read": True,
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "test_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "test_confusion_matrix.json").write_text(json.dumps(summary["confusion_matrix"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# 冻结模型最终 Test 评估报告

**状态**：`FINAL_TEST_EVALUATION_COMPLETE`  
**模型**：`B0_seed2029`  
**评估集**：附件2 `test`，共 {len(test)} 条  
**模型选择/调参**：未使用 test；本次只做一次冻结模型评估。  
**附件4结果**：未修改。  

## 指标

| 指标 | 值 |
|---|---:|
| Accuracy | {base_metrics['accuracy']:.6f} |
| Macro-F1 | {base_metrics['macro_f1']:.6f} |
| Negative F1 | {base_metrics['per_class_f1']['Negative']:.6f} |
| Neutral F1 | {base_metrics['per_class_f1']['Neutral']:.6f} |
| Positive F1 | {base_metrics['per_class_f1']['Positive']:.6f} |
| MAE | {base_metrics['mae']:.6f} |
| RMSE | {base_metrics['rmse']:.6f} |
| Pearson | {base_metrics['pearson'] if base_metrics['pearson'] is not None else 'NA'} |
| 选择指标 J（仅记录，不重新选择） | {base_metrics['selection_J']:.6f} |

混淆矩阵行列顺序均为 `Negative / Neutral / Positive`，详见 `test_confusion_matrix.json`。逐样本真值与预测详见本地 `test_predictions.csv`，不作为附件4解释结果发布。

## 约束核验

- 使用已冻结的 `B0_seed2029`，没有重新训练；
- scaler 的来源为 train；
- test 没有用于模型、架构、阈值或解释规则选择；
- 没有修改附件4预测、Shapley 或 Conditional IG 文件；
- 本报告是性能评估，不改变音频/视觉 `index_only` 的证据范围。
"""
    (out / "TEST_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {}
    for path in sorted(out.iterdir()):
        if path.name != "output_manifest.json" and path.is_file():
            manifest[path.name] = digest(path)
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "sample_count": len(test), "metrics": base_metrics, "test_used_for_selection": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
