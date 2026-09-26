"""Frozen-model Q3 valid-only explanation validation; final output is REVIEW_GATE."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

import numpy as np
from numpy._core.multiarray import _reconstruct
import torch

HERE = Path(__file__).resolve().parent
CORE = HERE if (HERE / "q3v1").is_dir() else HERE.parent
sys.path.insert(0, str(CORE))
from q3v1.data import EXPECTED_SHA256, TrainScaler, _validate_split, sha256_file
from q3v1.model import Q3Model
from q3v1.explain import conditional_ig, exact_shapley, local_attribution_status, summarize_contributions
from xai_checks import MODALITIES, fixed_target_ig, position_comparison, perturbation_rows, grouped_bootstrap


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=default, allow_nan=False) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload.encode("utf-8"))


def verify_sources(mode):
    manifest_path = HERE / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        if mode != "smoke":
            raise PermissionError("Formal run requires the verified frozen server bundle")
        return "DEVELOPMENT_SMOKE"
    manifest = read_json(manifest_path)
    for name, expected in manifest["files"].items():
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE) or sha256_file(path) != expected:
            raise ValueError("bundle source hash mismatch: " + name)
    return manifest["source_commit"]


def load_locked_inputs(args, config):
    assets = HERE / "assets"
    freeze = read_json(assets / "MODEL_FREEZE_MANIFEST.json")
    audit = read_json(assets / "audit_result.json")
    if freeze["selected_architecture"] != "B0" or freeze["delivery_seed"] != 2029 or audit["status"] != "PASS":
        raise ValueError("frozen training audit does not match selected B0")
    for name, expected in freeze["source_sha256"].items():
        if sha256_file(CORE / name) != expected:
            raise ValueError("frozen core source changed: " + name)
    if sha256_file(assets / "explanation_scope_contract.json") != freeze["explanation_scope_contract_sha256"]:
        raise ValueError("scope changed")
    frozen = read_json(assets / "frozen_config.json")
    for old, new in (("ig_steps_schedule", "ig_steps_schedule"), ("ig_completeness_atol", "ig_atol"),
                     ("ig_completeness_rtol", "ig_rtol"), ("attribution_epsilon_candidate", "epsilon"),
                     ("explanation_perturbation_fractions", "fractions"), ("random_controls_per_case", "random_controls")):
        if frozen[old] != config[new]:
            raise ValueError("validation configuration drift: " + new)
    hashes = audit["file_sha256"]
    for name in ("B0_seed2029.pt", "B0_seed2030.pt", "B0_seed2031.pt", "train_scaler.pt", "training_summary.json"):
        if sha256_file(args.run_dir / name) != hashes["b0b1_v1_cuda128/" + name]:
            raise ValueError("training artifact hash mismatch: " + name)
    if sha256_file(args.aligned_pkl) != EXPECTED_SHA256:
        raise ValueError("official aligned PKL hash mismatch")
    # The official pickle container deserializes all split bytes, but only the
    # train and valid mappings are accessed. No test evaluation or selection.
    with args.aligned_pkl.open("rb") as stream:
        source = pickle.load(stream)
    train = _validate_split(source["train"], "train")
    valid = _validate_split(source["valid"], "valid")
    del source
    if len(train) != 3395 or len(valid) != 728 or {s.sample_id for s in train} & {s.sample_id for s in valid}:
        raise ValueError("split counts or isolation failed")
    chosen = sorted(valid, key=lambda s: hashlib.sha256(("q3-xai-v1|" + s.sample_id).encode()).hexdigest())[:120]
    subset_bytes = ("\n".join(s.sample_id for s in chosen) + "\n").encode()
    if hashlib.sha256(subset_bytes).hexdigest() != config["subset_sha256"] or subset_bytes != (assets / "valid_xai_subset_120.txt").read_bytes():
        raise ValueError("frozen valid subset changed")
    if any("$_$" not in s.sample_id for s in chosen):
        raise ValueError("video_id extraction requires official sample ID delimiter")
    scaler = TrainScaler().fit(train, split="train")
    with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]):
        saved = torch.load(args.run_dir / "train_scaler.pt", map_location="cpu", weights_only=True)
    if saved["source_split"] != "train":
        raise ValueError("non-train scaler")
    for modality in MODALITIES:
        for stat in ("mean", "std"):
            if not np.array_equal(saved[stat][modality], getattr(scaler, stat)[modality]):
                raise ValueError("saved scaler differs from train refit")
    medians = {}
    for modality in MODALITIES:
        values = np.concatenate([getattr(s, modality)[s.content_indices] for s in train], axis=0)
        medians[modality] = np.median(values.astype(np.float64), axis=0)
        del values
    return chosen, scaler, medians, freeze


def load_model(run_dir, seed, device):
    ckpt = torch.load(run_dir / f"B0_seed{seed}.pt", map_location="cpu", weights_only=True)
    if ckpt["variant"] != "B0" or ckpt["seed"] != seed:
        raise ValueError("checkpoint identity differs")
    model = Q3Model("B0")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def prepare_sample(sample, scaler, medians, device):
    transformed = scaler.transform(sample)
    xs = tuple(torch.from_numpy(v[None]).to(device) for v in transformed[:3])
    mask = torch.from_numpy(transformed[3][None]).to(device)
    mean_ref = tuple(torch.zeros_like(x) for x in xs)
    median_ref = []
    for modality, x in zip(MODALITIES, xs):
        vector = ((medians[modality] - scaler.mean[modality]) / scaler.std[modality]).astype(np.float32)
        reference = torch.from_numpy(vector).to(device)[None, None, :].expand_as(x).clone()
        reference *= mask.unsqueeze(-1)
        median_ref.append(reference)
    return xs, mask, mean_ref, tuple(median_ref)


def core_ig(model, xs, ref, mask, modality, target, target_class, config):
    return conditional_ig(model, xs, ref, mask, modality=modality, target=target,
                          target_class=target_class if target == "classification" else None,
                          steps_schedule=tuple(config["ig_steps_schedule"]),
                          atol=config["ig_atol"], rtol=config["ig_rtol"])


def run(args):
    start = time.monotonic()
    source_commit = verify_sources(args.mode)
    config = read_json(HERE / "validation_contract.json")
    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.8.0" or np.__version__ != "2.5.0":
        raise ValueError("requires frozen Python3.12 / torch2.8.0 / numpy2.5.0")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("use a new empty output directory; no overwrite/resume")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")
    selected, scaler, medians, freeze = load_locked_inputs(args, config)
    planned = selected[:1] if args.mode == "smoke" else selected
    models = {seed: load_model(args.run_dir, seed, device) for seed in config["seed_stability"]}
    torch.manual_seed(config["seed"])
    randomized = Q3Model("B0").to(device).eval().requires_grad_(False)
    reference = models[2029]
    write_json(args.out / "run_identity.json", {
        "mode": args.mode, "source_commit": source_commit, "freeze": freeze,
        "config_sha256": sha256_file(HERE / "validation_contract.json"),
        "sample_ids": [s.sample_id for s in planned], "n_samples": len(planned),
        "device": str(device), "hostname": platform.node(), "python": platform.python_version(),
        "sys_executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "test_evaluated": False, "attachment4_inference": False, "optimizer_steps": 0})
    perturbations, sensitivities, numeric, reference_sensitivity = [], [], [], []
    local_statuses = []
    for ordinal, sample in enumerate(planned, 1):
        xs, mask, mean_ref, median_ref = prepare_sample(sample, scaler, medians, device)
        with torch.no_grad():
            logits, intensity = reference(*xs, mask)
            target_class = int(logits.argmax(-1).item())
        record = {"sample_id": sample.sample_id, "official_seq_indices": sample.content_indices,
                  "mapping_status": dict.fromkeys(MODALITIES, "index_only"),
                  "prediction": {"class": target_class, "intensity": float(intensity.item())}, "targets": {}}
        for target in ("classification", "regression"):
            sh = exact_shapley(reference, xs, mean_ref, mask, target=target)
            sh_median = exact_shapley(reference, xs, median_ref, mask, target=target)
            target_record = {"shapley_mean": sh, "shapley_median": sh_median,
                             "semantics": summarize_contributions(sh["phi"], target=target, epsilon=config["epsilon"]),
                             "modalities": {}}
            median_semantics = summarize_contributions(sh_median["phi"], target=target, epsilon=config["epsilon"])
            reference_sensitivity.append({"sample_id": sample.sample_id, "target": target,
                "phi_max_abs_shift": max(abs(sh["phi"][m] - sh_median["phi"][m]) for m in MODALITIES),
                "primary_influential_agreement": target_record["semantics"]["primary_influential_modality"] == median_semantics["primary_influential_modality"],
                "primary_supporting_agreement": (target_record["semantics"].get("primary_supporting_modality") == median_semantics.get("primary_supporting_modality")) if target == "classification" else None})
            for ref_name, value in (("mean", sh), ("median", sh_median)):
                numeric.append({"sample_id": sample.sample_id, "target": target, "method": "shapley",
                                "reference": ref_name, "pass": abs(value["additivity_residual"]) <= config["shapley_additivity_atol"],
                                "residual": value["additivity_residual"]})
            for modality in MODALITIES:
                mean_ig = core_ig(reference, xs, mean_ref, mask, modality, target, target_class, config)
                status = local_attribution_status(sh["phi"][modality], mean_ig, mask[0].cpu().numpy(), epsilon=config["epsilon"])
                local_statuses.append(status)
                comparisons = {"median_reference": core_ig(reference, xs, median_ref, mask, modality, target, target_class, config)}
                for seed in (2030, 2031):
                    comparisons[f"seed{seed}"] = fixed_target_ig(models[seed], xs, mean_ref, mask,
                        modality=modality, target=target, target_class=target_class, config=config)
                comparisons["randomized_model"] = fixed_target_ig(randomized, xs, mean_ref, mask,
                    modality=modality, target=target, target_class=target_class, config=config)
                for kind, ig in {"mean": mean_ig, **comparisons}.items():
                    scores = np.asarray(ig["position_scores"])
                    structural = np.isfinite(scores).all() and np.all(scores[~mask[0].cpu().numpy()] == 0)
                    numeric.append({"sample_id": sample.sample_id, "target": target, "modality": modality,
                                    "method": "conditional_ig", "context": kind, "steps": ig["steps"],
                                    "pass": ig["numerical_status"] == "pass" and bool(structural),
                                    "residual": ig["completeness_residual"]})
                if ordinal == 1 and target == "classification" and modality == "text":
                    repeated = core_ig(reference, xs, mean_ref, mask, modality, target, target_class, config)
                    adapter = fixed_target_ig(reference, xs, mean_ref, mask, modality=modality,
                                              target=target, target_class=target_class, config=config)
                    repeat_delta = float(np.max(np.abs(repeated["position_scores"] - mean_ig["position_scores"])))
                    adapter_delta = float(np.max(np.abs(adapter["position_scores"] - mean_ig["position_scores"])))
                    repeated_sh = exact_shapley(reference, xs, mean_ref, mask, target=target)
                    phi_delta = max(abs(sh["phi"][m] - repeated_sh["phi"][m]) for m in MODALITIES)
                    epsilon_ok = max(repeat_delta, adapter_delta, phi_delta) <= config["epsilon"]
                    write_json(args.out / "epsilon_precision_gate.json", {"epsilon": config["epsilon"],
                        "repeat_ig_max_abs_delta": repeat_delta, "fixed_target_adapter_max_abs_delta": adapter_delta,
                        "repeat_phi_max_abs_delta": phi_delta, "status": "PASS" if epsilon_ok else "FAIL"})
                    if not epsilon_ok:
                        raise RuntimeError("frozen epsilon/fixed-target equivalence failed; do not retune")
                rows = perturbation_rows(reference, xs, mean_ref, mask, mean_ig, status,
                                         sample.sample_id, target, target_class, modality, config)
                perturbations.extend(rows)
                target_record["modalities"][modality] = {"mean_ig": mean_ig, "local_status": status,
                                                        "diagnostics_ig": comparisons, "perturbations": rows}
                for kind, ig in comparisons.items():
                    sensitivities.append({"sample_id": sample.sample_id, "target": target,
                        "modality": modality, "comparison": kind, "mapping_status": "index_only",
                        **position_comparison(mean_ig, ig, sample.content_indices, config)})
            record["targets"][target] = target_record
        write_json(args.out / "samples" / f"sample_{ordinal:03d}.json", record)
        elapsed = time.monotonic() - start
        write_json(args.out / "progress.json", {"status": "RUNNING", "completed": ordinal,
                                               "total": len(planned), "elapsed_sec": elapsed})
        print(f"[{ordinal}/{len(planned)}] {sample.sample_id} elapsed={elapsed:.1f}s", flush=True)
    comparisons = {}
    for target in ("classification", "regression"):
        for modality in MODALITIES:
            for fraction in config["fractions"]:
                key = f"{target}/{modality}/{fraction}"
                rows = [r for r in perturbations if (r["target"], r["modality"], r["fraction"]) == (target, modality, fraction)]
                comparisons[key] = grouped_bootstrap(rows, config, key)
    failures = [r for r in numeric if not r["pass"]]
    sensitivity_summary = {}
    for target in ("classification", "regression"):
        for modality in MODALITIES:
            for kind in ("median_reference", "seed2030", "seed2031", "randomized_model"):
                rows = [r for r in sensitivities if (r["target"], r["modality"], r["comparison"]) == (target, modality, kind)]
                eligible = [r for r in rows if r["status"] == "ok"]
                sensitivity_summary[f"{target}/{modality}/{kind}"] = {
                    "n_total": len(rows), "n_eligible": len(eligible),
                    "status_counts": dict(Counter(r["status"] for r in rows)),
                    "mean_spearman_eligible": float(np.mean([r["spearman"] for r in eligible])) if eligible else None,
                    "mean_topk_overlap_eligible": {str(f): float(np.mean([r["overlap"][str(f)] for r in eligible])) if eligible else None
                                                   for f in config["fractions"]}}
    write_json(args.out / "numeric_checks.json", numeric)
    write_json(args.out / "sensitivity_checks.json", sensitivities)
    write_json(args.out / "reference_sensitivity.json", reference_sensitivity)
    write_json(args.out / "sensitivity_summary.json", sensitivity_summary)
    write_json(args.out / "perturbation_summary.json", comparisons)
    summary = {"status": "SMOKE_COMPLETE" if args.mode == "smoke" else "FORMAL_XAI_COMPLETE",
               "final_gate": "REVIEW_GATE", "sample_count": len(planned),
               "numerical_status": "PASS" if not failures else "REVIEW_REQUIRED",
               "numerical_checks": len(numeric), "numerical_failures": failures,
               "local_status_counts": dict(Counter(local_statuses)),
               "sensitivity_summary": sensitivity_summary,
               "perturbation_comparisons": comparisons, "effectiveness_gate": "REQUIRES_EVIDENCE_REVIEW",
               "test_evaluated": False, "attachment4_inference": False, "elapsed_sec": time.monotonic() - start}
    write_json(args.out / "xai_validation_summary.json", summary)
    write_json(args.out / "progress.json", {"status": summary["status"], "completed": len(planned), "total": len(planned)})
    write_json(args.out / "output_manifest.json", {p.relative_to(args.out).as_posix(): sha256_file(p)
               for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "output_manifest.json"})
    print(json.dumps({"status": summary["status"], "numerical_status": summary["numerical_status"],
                      "final_gate": "REVIEW_GATE", "elapsed_sec": summary["elapsed_sec"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-pkl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    # Avoid changing an existing run even when admission fails.
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("output directory must be empty")
    try:
        run(args)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "failure.json", {"status": "FAIL_STOP", "type": type(exc).__name__,
                                             "message": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
