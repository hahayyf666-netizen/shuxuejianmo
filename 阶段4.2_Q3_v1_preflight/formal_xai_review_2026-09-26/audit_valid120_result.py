"""Read-only structural and statistical audit of the frozen valid120 XAI output."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition: bool, message: str, failures: list[str]):
    if not condition:
        failures.append(message)


def rng(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "big")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    failures = []
    archive_sha = sha256(args.archive)
    expected_sha = "33ef931cfacfee17ce36746b975e4d0440a7aec5371ddb616ad32193876b6645"
    with ZipFile(args.archive) as bundle:
        names = bundle.namelist()
        bad_member = bundle.testzip()
    check(archive_sha == expected_sha, "archive SHA differs from server receipt", failures)
    check(len(names) == len(set(names)) == 132 and bad_member is None, "ZIP count/CRC/duplicate failure", failures)
    check(all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names), "unsafe ZIP name", failures)

    run = args.extracted / "xai_v1_cuda128"
    identity, summary, progress = (read(run / n) for n in
                                  ("run_identity.json", "xai_validation_summary.json", "progress.json"))
    config = read(args.contract)
    subset = args.subset.read_text(encoding="utf-8").splitlines()
    check(identity["sample_ids"] == subset and len(subset) == len(set(subset)) == 120,
          "frozen valid120 list differs", failures)
    check(identity["source_commit"] == "a6b6bb552613cd4c59ab23e99f15b5b8a920a082", "source commit differs", failures)
    check(identity["config_sha256"] == sha256(args.contract), "contract hash differs", failures)
    check(identity["freeze"]["selected_architecture"] == "B0" and identity["freeze"]["delivery_seed"] == 2029,
          "selected model differs", failures)
    check(identity["freeze"]["aligned_50_pkl_sha256"] ==
          "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd", "dataset hash differs", failures)
    check(identity["mode"] == "formal" and identity["device"] == "cuda:1" and
          identity["optimizer_steps"] == 0 and not identity["test_evaluated"] and
          not identity["attachment4_inference"], "run mode/boundary differs", failures)
    check(summary["status"] == progress["status"] == "FORMAL_XAI_COMPLETE" and
          summary["final_gate"] == "REVIEW_GATE" and summary["sample_count"] == progress["completed"] == 120 and
          summary["numerical_status"] == "PASS" and not summary["numerical_failures"], "completion gate differs", failures)
    manifest = read(run / "output_manifest.json")
    actual = {p.relative_to(run).as_posix(): sha256(p) for p in run.rglob("*")
              if p.is_file() and p.name != "output_manifest.json"}
    check(manifest == actual, "output manifest does not match files", failures)

    numeric = read(run / "numeric_checks.json")
    check(len(numeric) == summary["numerical_checks"] == 4080 and all(x["pass"] and math.isfinite(x["residual"])
          for x in numeric), "numeric check count or values differ", failures)
    check(read(run / "epsilon_precision_gate.json")["status"] == "PASS", "epsilon gate differs", failures)
    stdout = (args.extracted / "xai_v1_cuda128.stdout.log").read_text(encoding="utf-8")
    stderr = (args.extracted / "xai_v1_cuda128.stderr.log").read_text(encoding="utf-8")
    check(stdout.count(" elapsed=") == 120 and not stderr.strip(), "run log count or stderr differs", failures)

    rows = defaultdict(list)
    local_status = Counter()
    primary, support = defaultdict(Counter), Counter()
    positive_support = defaultdict(list)
    reference_rows = read(run / "reference_sensitivity.json")
    sample_files = sorted((run / "samples").glob("sample_*.json"))
    check(len(sample_files) == 120, "sample JSON count differs", failures)
    for ordinal, path in enumerate(sample_files):
        sample = read(path)
        sid = sample["sample_id"]
        check(ordinal < len(subset) and sid == subset[ordinal], f"sample ordinal mismatch: {path.name}", failures)
        indices = sample["official_seq_indices"]
        check(indices == sorted(set(indices)) and 0 < len(indices) <= 50 and all(0 <= x < 50 for x in indices),
              f"invalid content indices: {sid}", failures)
        check(sample["mapping_status"] == dict.fromkeys(("text", "audio", "vision"), "index_only"),
              f"mapping status differs: {sid}", failures)
        for target in ("classification", "regression"):
            data = sample["targets"][target]
            primary[target][data["semantics"]["primary_influential_modality"]] += 1
            if target == "classification":
                support[data["semantics"]["primary_supporting_modality"]] += 1
            for reference in ("shapley_mean", "shapley_median"):
                sh = data[reference]
                delta = sh["coalition_values"]["7"] - sh["coalition_values"]["0"]
                check(abs(sum(sh["phi"].values()) - delta) <= config["shapley_additivity_atol"] + 1e-8,
                      f"Shapley additivity differs: {sid}/{target}/{reference}", failures)
                check(sh["target_class"] == (sample["prediction"]["class"] if target == "classification" else None),
                      f"fixed class target differs: {sid}/{target}/{reference}", failures)
            ref_row = reference_rows[2 * ordinal + (0 if target == "classification" else 1)]
            sh0, sh1 = data["shapley_mean"], data["shapley_median"]
            max_shift = max(abs(sh0["phi"][m] - sh1["phi"][m]) for m in ("text", "audio", "vision"))
            check(ref_row["sample_id"] == sid and ref_row["target"] == target and
                  abs(ref_row["phi_max_abs_shift"] - max_shift) < 1e-12,
                  f"reference sensitivity differs: {sid}/{target}", failures)
            for modality in ("text", "audio", "vision"):
                part = data["modalities"][modality]
                local_status[part["local_status"]] += 1
                ig = part["mean_ig"]
                scores = ig["position_scores"]
                check(len(scores) == 50 and ig["numerical_status"] == "pass" and
                      all(math.isfinite(v) for v in scores) and
                      abs(sum(scores) - ig["conditional_output_difference"] - ig["completeness_residual"]) < 1e-5 and
                      all(abs(scores[i]) < 1e-8 for i in range(50) if i not in indices),
                      f"IG structure/completeness differs: {sid}/{target}/{modality}", failures)
                check(ig["target_class"] == (sample["prediction"]["class"] if target == "classification" else None),
                      f"IG fixed class target differs: {sid}/{target}/{modality}", failures)
                check(len(part["perturbations"]) == 3, f"fraction count differs: {sid}/{target}/{modality}", failures)
                for row in part["perturbations"]:
                    fraction = row["fraction"]
                    key = f"{target}/{modality}/{fraction}"
                    rows[key].append(row)
                    k = max(1, math.ceil(fraction * len(indices)))
                    check(row["k"] == k and row["valid_position_count"] == len(indices) and
                          row["sample_id"] == sid and row["video_id"] == sid.rsplit("$_$", 1)[0] and
                          row["mapping_status"] == "index_only" and
                          row["target_class"] == (sample["prediction"]["class"] if target == "classification" else None),
                          f"perturbation identity differs: {sid}/{key}", failures)
                    if row["eligible"]:
                        top = row["top_positions"]
                        random_sets = row["random_positions"]
                        random_values = row["random_abs_changes"]
                        check(len(top) == k and set(top) <= set(indices) and len(set(top)) == k and
                              len(random_sets) == len(random_values) == 50 and
                              all(len(set(r)) == k and set(r) <= set(indices) for r in random_sets),
                              f"perturbation position selection differs: {sid}/{key}", failures)
                        # The server computes both expressions in float32 before JSON serialization.
                        check(float(np.asarray(random_values, dtype=np.float32).mean()) == row["random_mean_abs_change"] and
                              float(np.float32(row["top_abs_change"]) - np.float32(row["random_mean_abs_change"])) == row["paired_difference"],
                              f"perturbation difference differs: {sid}/{key}", failures)
                        if target == "classification" and row["positive_support_signed_drop"] is not None:
                            positive_support[f"{modality}/{fraction}"].append(row["positive_support_signed_drop"])

    check(dict(local_status) == summary["local_status_counts"], "local status tally differs", failures)
    comparisons = read(run / "perturbation_summary.json")
    bootstrap_max_abs_error = 0.0
    for key, bucket in rows.items():
        item = comparisons[key]
        eligible = [r for r in bucket if r["eligible"]]
        groups = sorted({r["video_id"] for r in eligible})
        check(item["n_total"] == 120 and item["n_eligible"] == len(eligible) and
              item["n_video_groups"] == len(groups), f"bootstrap denominator differs: {key}", failures)
        diffs = np.asarray([r["paired_difference"] for r in eligible])
        bootstrap_max_abs_error = max(bootstrap_max_abs_error, abs(float(diffs.mean()) - item["mean_paired_difference"]))
        sums = np.asarray([sum(r["paired_difference"] for r in eligible if r["video_id"] == g) for g in groups])
        counts = np.asarray([sum(r["video_id"] == g for r in eligible) for g in groups])
        draw = rng(config["seed"], "bootstrap", key).integers(0, len(groups), size=(2000, len(groups)))
        ci = np.quantile(sums[draw].sum(1) / counts[draw].sum(1), [0.025, 0.975])
        bootstrap_max_abs_error = max(bootstrap_max_abs_error, *np.abs(ci - item["ci95"]))
        check(np.allclose(ci, item["ci95"], atol=1e-12, rtol=0), f"bootstrap CI differs: {key}", failures)
        check(item["evidence"] == ("above_random" if ci[0] > 0 else "below_random" if ci[1] < 0 else "inconclusive"),
              f"bootstrap evidence label differs: {key}", failures)

    sensitivity = read(run / "sensitivity_checks.json")
    sensitivity_summary = read(run / "sensitivity_summary.json")
    check(len(sensitivity) == 2880 and len(sensitivity_summary) == 24, "sensitivity count differs", failures)
    for key, aggregate in sensitivity_summary.items():
        target, modality, comparison = key.split("/")
        bucket = [r for r in sensitivity if (r["target"], r["modality"], r["comparison"]) ==
                  (target, modality, comparison)]
        eligible = [r for r in bucket if r["status"] == "ok"]
        check(aggregate["n_total"] == len(bucket) == 120 and aggregate["n_eligible"] == len(eligible) and
              aggregate["status_counts"] == dict(Counter(r["status"] for r in bucket)),
              f"sensitivity denominator differs: {key}", failures)
        if eligible:
            check(abs(np.mean([r["spearman"] for r in eligible]) - aggregate["mean_spearman_eligible"]) < 1e-12,
                  f"sensitivity average differs: {key}", failures)
            for fraction in config["fractions"]:
                f = str(fraction)
                mean_overlap = np.mean([r["overlap"][f] for r in eligible])
                check(abs(mean_overlap - aggregate["mean_topk_overlap_eligible"][f]) < 1e-12,
                      f"sensitivity overlap differs: {key}/{f}", failures)

    result = {
        "status": "PASS" if not failures else "FAIL",
        "archive_sha256": archive_sha,
        "source_commit": identity["source_commit"],
        "sample_count": len(sample_files), "video_groups": len({r["video_id"] for r in rows["classification/text/0.1"]}),
        "output_manifest_files": len(manifest), "numeric_checks": len(numeric),
        "perturbation_comparisons": len(comparisons),
        "perturbation_above_random_count": sum(v["evidence"] == "above_random" for v in comparisons.values()),
        "bootstrap_max_abs_error": bootstrap_max_abs_error,
        "local_status_counts": dict(local_status),
        "primary_influential_counts": {k: dict(v) for k, v in primary.items()},
        "classification_primary_supporting_counts": dict(support),
        "positive_support_signed_drop": {k: {"n": len(v), "mean": float(np.mean(v)),
                                               "fraction_positive": float(np.mean(np.asarray(v) > 0))}
                                         for k, v in positive_support.items()},
        "reference_sensitivity_rows": len(reference_rows),
        "sensitivity_rows": len(sensitivity),
        "stderr_empty": not stderr.strip(),
        "formal_valid120_run": True, "test_evaluated": False, "attachment4_inference": False,
        "final_gate": "REVIEW_GATE", "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": len(failures),
                      "archive_sha256": archive_sha, "sample_count": result["sample_count"],
                      "above_random": result["perturbation_above_random_count"]}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
