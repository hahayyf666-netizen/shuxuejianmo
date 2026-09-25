"""Fixed-reference model attribution for the frozen three-modality game."""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import torch

MODALITIES = ("text", "audio", "vision")


def _target(model, xs: tuple[torch.Tensor, ...], mask: torch.Tensor, target: str, target_class: int) -> torch.Tensor:
    logits, intensity = model(*xs, mask)
    if target == "classification":
        return torch.softmax(logits, dim=-1)[:, target_class]
    if target == "regression":
        return intensity
    raise ValueError("target must be classification or regression")


def _validate(xs: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...], mask: torch.Tensor) -> None:
    if len(xs) != 3 or len(reference) != 3 or xs[0].shape[0] != 1:
        raise ValueError("one sample and three modality tensors required")
    if any(x.shape != b.shape for x, b in zip(xs, reference)):
        raise ValueError("reference shape mismatch")
    if mask.shape != (1, 50) or mask.dtype != torch.bool:
        raise ValueError("invalid mask")


def exact_shapley(
    model, xs: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...], mask: torch.Tensor,
    *, target: str, target_class: int | None = None,
) -> dict:
    """Enumerate all 8 coalitions. Class target is frozen from the full input."""
    _validate(xs, reference, mask)
    model.eval()
    with torch.no_grad():
        if target == "classification":
            full_logits, _ = model(*xs, mask)
            predicted = int(full_logits.argmax(dim=-1).item())
            if target_class is None:
                target_class = predicted
            elif target_class != predicted:
                raise ValueError("class explanation must target full-input predicted class")
        else:
            target_class = 0
        values: dict[int, float] = {}
        for bits in range(8):
            coalition = tuple(xs[i] if bits & (1 << i) else reference[i] for i in range(3))
            values[bits] = float(_target(model, coalition, mask, target, target_class).item())
    phi = {}
    for i, name in enumerate(MODALITIES):
        others = [j for j in range(3) if j != i]
        value = 0.0
        for k in range(3):
            for subset in combinations(others, k):
                bits = sum(1 << j for j in subset)
                weight = math.factorial(k) * math.factorial(2 - k) / math.factorial(3)
                value += weight * (values[bits | (1 << i)] - values[bits])
        phi[name] = value
    residual = sum(phi.values()) - (values[7] - values[0])
    return {"target": target, "target_class": target_class if target == "classification" else None,
            "phi": phi, "coalition_values": values, "additivity_residual": residual}


def conditional_ig(
    model, xs: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...], mask: torch.Tensor,
    *, modality: str, target: str, target_class: int | None = None,
    steps_schedule: tuple[int, ...] = (64, 128, 256), atol: float = 1e-3, rtol: float = 0.01,
) -> dict:
    """Only one modality moves; the other two remain at their actual values."""
    _validate(xs, reference, mask)
    if modality not in MODALITIES:
        raise ValueError("unknown modality")
    model.eval()
    i = MODALITIES.index(modality)
    with torch.no_grad():
        if target == "classification":
            pred = int(model(*xs, mask)[0].argmax(dim=-1).item())
            if target_class is None:
                target_class = pred
            elif target_class != pred:
                raise ValueError("class target must be full-input predicted class")
        else:
            target_class = 0
        ablated = list(xs)
        ablated[i] = reference[i]
        f_full = float(_target(model, xs, mask, target, target_class).item())
        f_ablated = float(_target(model, tuple(ablated), mask, target, target_class).item())
    delta = xs[i] - reference[i]
    result = None
    for steps in steps_schedule:
        nodes, weights = np.polynomial.legendre.leggauss(steps)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        gradient_sum = torch.zeros_like(xs[i])
        for alpha, weight in zip(nodes, weights):
            moving = (reference[i] + float(alpha) * delta).detach().requires_grad_(True)
            inputs = list(xs)
            inputs[i] = moving
            output = _target(model, tuple(inputs), mask, target, target_class)
            grad = torch.autograd.grad(output.sum(), moving)[0]
            gradient_sum += float(weight) * grad.detach()
        attrs = delta * gradient_sum
        attrs = attrs * mask.unsqueeze(-1)
        position_scores = attrs.sum(dim=-1).squeeze(0)
        residual = float(position_scores.sum().item()) - (f_full - f_ablated)
        passed = abs(residual) <= atol + rtol * abs(f_full - f_ablated)
        result = {"modality": modality, "target": target,
                  "target_class": target_class if target == "classification" else None,
                  "position_scores": position_scores.detach().cpu().numpy(),
                  "conditional_output_difference": f_full - f_ablated,
                  "completeness_residual": residual, "steps": steps,
                  "numerical_status": "pass" if passed else "fail"}
        if passed:
            break
    return result


def summarize_contributions(phi: dict[str, float], *, target: str, epsilon: float = 1e-6) -> dict:
    if set(phi) != set(MODALITIES) or epsilon <= 0 or any(not np.isfinite(v) for v in phi.values()):
        raise ValueError("invalid contributions or tolerance")
    max_abs = max(abs(v) for v in phi.values())
    influential_ties = [name for name in MODALITIES if abs(abs(phi[name]) - max_abs) <= epsilon]
    influential = "unresolved" if max_abs <= epsilon else influential_ties[0]
    denominator = sum(abs(v) for v in phi.values())
    result = {"primary_influential_modality": influential, "influential_ties": influential_ties,
              "relative_magnitude": {k: abs(v) / denominator if denominator > epsilon else 0.0 for k, v in phi.items()}}
    if target == "classification":
        positives = {k: v for k, v in phi.items() if v > epsilon}
        result["primary_supporting_modality"] = max(positives, key=positives.get) if positives else "NONE"
        if positives:
            peak = max(positives.values())
            result["supporting_ties"] = [k for k in MODALITIES if k in positives and abs(phi[k] - peak) <= epsilon]
        else:
            result["supporting_ties"] = []
    elif target == "regression":
        result["direction_labels"] = {k: "push_positive" if v > epsilon else "push_negative" if v < -epsilon else "neutral_relative_to_reference" for k, v in phi.items()}
    else:
        raise ValueError("unknown target")
    return result


def local_attribution_status(phi_value: float, ig: dict, mask: np.ndarray, *, epsilon: float = 1e-6) -> str:
    scores = np.asarray(ig["position_scores"])[np.asarray(mask, dtype=bool)]
    if ig["numerical_status"] != "pass":
        return "local_attribution_numerical_failure"
    if abs(phi_value) > epsilon and (not len(scores) or float(np.abs(scores).sum()) <= epsilon):
        return "local_attribution_unresolved_interaction"
    if not len(scores) or float(np.abs(scores).sum()) <= epsilon:
        return "local_attribution_weak"
    return "localized_feature_position"
