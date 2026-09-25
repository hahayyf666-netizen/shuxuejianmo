"""CPU-only contract and numeric smoke. No optimizer step, test metrics or Attachment4 prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
import torch

from q3v1.data import EXPECTED_SHA256, TrainScaler, _validate_split, sha256_file
from q3v1.explain import conditional_ig, exact_shapley, local_attribution_status, summarize_contributions
from q3v1.model import Q3Model


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-pkl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    actual_sha = sha256_file(args.aligned_pkl)
    if actual_sha != EXPECTED_SHA256:
        raise ValueError("Attachment2 aligned SHA-256 mismatch")
    with args.aligned_pkl.open("rb") as stream:
        container = pickle.load(stream)
    # Python pickle deserializes the container, including test bytes, but this
    # program never indexes container['test'] or computes test metrics.
    train = _validate_split(container["train"], "train")
    valid = _validate_split(container["valid"], "valid")
    del container
    if len(train) != 3395 or len(valid) != 728 or set(s.sample_id for s in train) & set(s.sample_id for s in valid):
        raise ValueError("official split count or isolation mismatch")
    scaler = TrainScaler().fit(train, split="train")
    np.savez_compressed(args.out / "train_scaler.npz", **{
        f"{modality}_{stat}": getattr(scaler, stat)[modality].astype(np.float64)
        for modality in ("text", "audio", "vision") for stat in ("mean", "std")
    })

    sample = train[0]
    transformed = scaler.transform(sample)
    xs = tuple(torch.from_numpy(x[None, :, :]) for x in transformed[:3])
    mask = torch.from_numpy(transformed[3][None, :])
    model_smoke = {}
    for variant in ("B0", "B1"):
        torch.manual_seed(2029)
        model = Q3Model(variant)
        if model.parameter_count > 1_000_000:
            raise ValueError("parameter budget exceeded")
        model.eval()
        logits, intensity = model(*xs, mask)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([sample.classification_label])) + \
               torch.abs(intensity - torch.tensor([sample.regression_label], dtype=torch.float32)).mean()
        loss.backward()  # no optimizer step
        model_smoke[variant] = {
            "parameter_count": model.parameter_count,
            "fp32_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "logits_shape": list(logits.shape),
            "intensity_in_range": bool(torch.all((intensity >= -3) & (intensity <= 3))),
            "loss_finite": bool(torch.isfinite(loss)),
            "backward_gradients_present": any(p.grad is not None for p in model.parameters()),
            "optimizer_steps": 0,
        }

    # Untrained B0 numbers establish implementation/numeric behavior only.
    torch.manual_seed(2029)
    explain_model = Q3Model("B0").eval()
    ref = tuple(torch.zeros_like(x) for x in xs)  # train mean in standardized coordinates
    cls_shapley = exact_shapley(explain_model, xs, ref, mask, target="classification")
    reg_shapley = exact_shapley(explain_model, xs, ref, mask, target="regression")
    ig = conditional_ig(explain_model, xs, ref, mask, modality="text", target="classification")
    selected_valid = sorted(valid, key=lambda s: digest_bytes(("q3-xai-v1|" + s.sample_id).encode()))[:120]
    subset_bytes = "\n".join(s.sample_id for s in selected_valid).encode("utf-8")
    (args.out / "valid_xai_subset_120.txt").write_bytes(subset_bytes + b"\n")

    report = {
        "status": "preflight_implementation_smoke_pass_mapping_gate_open",
        "aligned_pkl_sha256": actual_sha,
        "split_counts": {"train": len(train), "valid": len(valid), "test_accessed_for_selection": False},
        "train_valid_id_overlap": 0,
        "content_position_lengths": {
            name: {"min": min(len(s.content_indices) for s in samples),
                   "max": max(len(s.content_indices) for s in samples),
                   "mean": float(np.mean([len(s.content_indices) for s in samples]))}
            for name, samples in (("train", train), ("valid", valid))
        },
        "scaler": {"source_split": "train", "small_std_dims": scaler.small_std,
                   "file_sha256": sha256_file(args.out / "train_scaler.npz")},
        "smoke_sample_id": sample.sample_id,
        "models": model_smoke,
        "attribution_numeric_smoke": {
            "trained_model": False,
            "classification_shapley_residual": cls_shapley["additivity_residual"],
            "regression_shapley_residual": reg_shapley["additivity_residual"],
            "classification_semantics": summarize_contributions(cls_shapley["phi"], target="classification"),
            "ig_steps": ig["steps"], "ig_completeness_residual": ig["completeness_residual"],
            "ig_numerical_status": ig["numerical_status"],
            "ig_local_status": local_attribution_status(cls_shapley["phi"]["text"], ig, mask[0].numpy()),
        },
        "valid_xai_subset_120_sha256": digest_bytes(subset_bytes + b"\n"),
        "environment": {"hostname": platform.node(), "python_executable": sys.executable,
                        "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available()},
        "explicit_non_actions": ["no training optimizer step", "no test metrics or selection", "no Attachment4 prediction"],
    }
    (args.out / "preflight_machine_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "split_counts": report["split_counts"],
                      "models": model_smoke, "ig_status": ig["numerical_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
