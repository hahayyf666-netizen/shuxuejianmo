"""Feature-space diagnostics; the frozen model and primary explain.py stay unchanged."""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch

MODALITIES = ("text", "audio", "vision")


def stable_rng(*parts):
    payload = "|".join(map(str, parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(seed))


def target_value(model, xs, mask, target, target_class):
    logits, regression = model(*xs, mask)
    if target == "classification":
        return logits.softmax(-1)[:, target_class]
    if target == "regression":
        return regression
    raise ValueError("invalid explanation target")


def fixed_target_ig(model, xs, reference, mask, *, modality, target, target_class, config):
    """Same conditional IG integral, allowing c* from the original model.

    Only used for seed stability and full-parameter randomization, where the
    comparison network may predict a different class. Never relabel its target.
    """
    model.eval()
    i = MODALITIES.index(modality)
    with torch.no_grad():
        base = list(xs)
        base[i] = reference[i]
        difference = float((target_value(model, xs, mask, target, target_class) -
                            target_value(model, tuple(base), mask, target, target_class)).item())
    delta = xs[i] - reference[i]
    for steps in config["ig_steps_schedule"]:
        nodes, weights = np.polynomial.legendre.leggauss(steps)
        total = torch.zeros_like(xs[i])
        for alpha, weight in zip((nodes + 1) / 2, weights / 2):
            moving = (reference[i] + float(alpha) * delta).detach().requires_grad_(True)
            current = list(xs)
            current[i] = moving
            output = target_value(model, tuple(current), mask, target, target_class)
            gradient = torch.autograd.grad(output.sum(), moving)[0]
            total += float(weight) * gradient.detach()
        scores = (delta * total * mask.unsqueeze(-1)).sum(-1).squeeze(0)
        residual = float(scores.sum().item()) - difference
        passed = abs(residual) <= config["ig_atol"] + config["ig_rtol"] * abs(difference)
        result = {"modality": modality, "target": target,
                  "target_class": target_class if target == "classification" else None,
                  "position_scores": scores.detach().cpu().numpy(),
                  "conditional_output_difference": difference,
                  "completeness_residual": residual, "steps": steps,
                  "numerical_status": "pass" if passed else "fail"}
        if passed:
            break
    return result


def rank_positions(scores, valid_indices):
    indices = np.asarray(valid_indices, dtype=np.int64)
    values = np.asarray(scores)
    if not np.isfinite(values).all() or indices.size == 0:
        raise ValueError("invalid attribution scores/indices")
    return indices[np.lexsort((indices, -np.abs(values[indices])))]


def average_ranks(values):
    values = np.asarray(values)
    ranks = np.zeros(values.shape, dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks[:] = (starts + (counts - 1) / 2)[inverse]
    return ranks


def position_comparison(left, right, indices, config):
    if left["numerical_status"] != "pass" or right["numerical_status"] != "pass":
        return {"status": "numerical_failure", "spearman": None, "overlap": {}}
    a = np.abs(np.asarray(left["position_scores"])[indices])
    b = np.abs(np.asarray(right["position_scores"])[indices])
    if a.sum() <= config["epsilon"] or b.sum() <= config["epsilon"]:
        return {"status": "weak_or_unlocalized", "spearman": None, "overlap": {}}
    ra, rb = average_ranks(a), average_ranks(b)
    corr = float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else None
    ar = rank_positions(left["position_scores"], indices)
    br = rank_positions(right["position_scores"], indices)
    overlap = {}
    for fraction in config["fractions"]:
        k = max(1, math.ceil(fraction * len(indices)))
        overlap[str(fraction)] = len(set(ar[:k]) & set(br[:k])) / k
    return {"status": "ok" if corr is not None else "constant_ranks",
            "spearman": corr, "overlap": overlap}


@torch.no_grad()
def replace_and_evaluate(model, xs, reference, mask, modality, subsets, target, target_class):
    i = MODALITIES.index(modality)
    batch_size = len(subsets)
    batch = [x.repeat(batch_size, 1, 1) for x in xs]
    for row, positions in enumerate(subsets):
        if len(positions):
            batch[i][row, positions, :] = reference[i][0, positions, :]
    return target_value(model, tuple(batch), mask.repeat(batch_size, 1), target, target_class).cpu().numpy()


def perturbation_rows(model, xs, reference, mask, ig, local_status, sample_id, target, target_class, modality, config):
    indices = np.flatnonzero(mask[0].detach().cpu().numpy())
    with torch.no_grad():
        full = float(target_value(model, xs, mask, target, target_class).item())
    scores = np.asarray(ig["position_scores"])
    rows = []
    for fraction in config["fractions"]:
        k = max(1, math.ceil(fraction * len(indices)))
        row = {"sample_id": sample_id, "video_id": sample_id.rsplit("$_$", 1)[0],
               "target": target, "target_class": target_class if target == "classification" else None,
               "modality": modality, "fraction": fraction, "k": k,
               "valid_position_count": len(indices), "mapping_status": "index_only",
               "local_status": local_status, "eligible": local_status == "localized_feature_position"}
        if not row["eligible"]:
            rows.append(row)
            continue
        top = rank_positions(scores, indices)[:k]
        rng = stable_rng(config["seed"], sample_id, target, modality, fraction)
        random_sets = [rng.choice(indices, size=k, replace=False) for _ in range(config["random_controls"])]
        changed = replace_and_evaluate(model, xs, reference, mask, modality,
                                       [top] + random_sets, target, target_class)
        absolute_changes = np.abs(changed - full)
        row.update({"top_positions": top.tolist(), "full_target": full,
                    "top_abs_change": float(absolute_changes[0]),
                    "top_signed_drop": float(full - changed[0]),
                    "random_abs_changes": absolute_changes[1:].tolist(),
                    "random_positions": [r.tolist() for r in random_sets],
                    "random_mean_abs_change": float(absolute_changes[1:].mean()),
                    "paired_difference": float(absolute_changes[0] - absolute_changes[1:].mean())})
        if target == "classification":
            positive = indices[scores[indices] > config["epsilon"]]
            positive = positive[np.lexsort((positive, -scores[positive]))][:k]
            drop = None
            if len(positive):
                result = replace_and_evaluate(model, xs, reference, mask, modality,
                                              [positive], target, target_class)[0]
                drop = float(full - result)
            row.update({"positive_support_positions": positive.tolist(),
                        "positive_support_count": len(positive), "positive_support_signed_drop": drop})
        rows.append(row)
    return rows


def grouped_bootstrap(rows, config, key):
    eligible = [r for r in rows if r["eligible"]]
    groups = sorted({r["video_id"] for r in eligible})
    result = {"n_total": len(rows), "n_eligible": len(eligible), "n_skipped": len(rows) - len(eligible),
              "n_video_groups": len(groups), "mean_paired_difference": None, "ci95": None,
              "evidence": "insufficient_groups"}
    if not eligible:
        return result
    diffs = np.asarray([r["paired_difference"] for r in eligible])
    result["mean_paired_difference"] = float(diffs.mean())
    if len(groups) < 2:
        return result
    sums = np.asarray([sum(r["paired_difference"] for r in eligible if r["video_id"] == g) for g in groups])
    counts = np.asarray([sum(r["video_id"] == g for r in eligible) for g in groups])
    rng = stable_rng(config["seed"], "bootstrap", key)
    draws = rng.integers(0, len(groups), size=(config["bootstrap_replicates"], len(groups)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    low, high = np.quantile(estimates, config["bootstrap_ci"])
    result["ci95"] = [float(low), float(high)]
    result["evidence"] = "above_random" if low > 0 else "below_random" if high < 0 else "inconclusive"
    return result
