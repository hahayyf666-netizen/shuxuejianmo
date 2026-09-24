from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fixed-sample Q1 NPZ arrays against formal production.")
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reproduction-root", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    reproduction_root = args.reproduction_root.resolve()
    reference_rows = {r["sample_key"]: r for r in read_rows(reference_root / "manifest.csv")}
    reproduced_rows = {r["sample_key"]: r for r in read_rows(reproduction_root / "manifest.csv")}
    extra = set(reproduced_rows).difference(reference_rows)
    if extra:
        raise SystemExit(f"reproduction contains keys absent from reference: {sorted(extra)}")

    sample_reports = []
    for sample_key in sorted(reproduced_rows):
        reference_path = reference_root / reference_rows[sample_key]["output_file"]
        reproduction_path = reproduction_root / reproduced_rows[sample_key]["output_file"]
        sample_report = {"sample_key": sample_key, "reference_file": reference_rows[sample_key]["output_file"],
                         "reproduction_file": reproduced_rows[sample_key]["output_file"], "field_count": 0,
                         "passed": True, "field_differences": []}
        with np.load(reference_path, allow_pickle=False) as ref, np.load(reproduction_path, allow_pickle=False) as rep:
            if set(ref.files) != set(rep.files):
                sample_report["passed"] = False
                sample_report["field_differences"].append({"field": "__field_set__", "error": "archive field sets differ"})
            else:
                sample_report["field_count"] = len(ref.files)
                for name in ref.files:
                    a, b = ref[name], rep[name]
                    if a.shape != b.shape or a.dtype != b.dtype:
                        sample_report["passed"] = False
                        sample_report["field_differences"].append({"field": name, "error": "shape or dtype differs", "reference_shape": list(a.shape), "reproduction_shape": list(b.shape), "reference_dtype": str(a.dtype), "reproduction_dtype": str(b.dtype)})
                    elif np.issubdtype(a.dtype, np.number):
                        equal = np.array_equal(a, b) if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_ else np.allclose(a, b, rtol=args.rtol, atol=args.atol, equal_nan=True)
                        if not equal:
                            finite_pair = np.isfinite(a) & np.isfinite(b) if np.issubdtype(a.dtype, np.floating) else np.ones(a.shape, dtype=bool)
                            max_abs = float(np.max(np.abs(a[finite_pair] - b[finite_pair]))) if finite_pair.any() else None
                            sample_report["passed"] = False
                            sample_report["field_differences"].append({"field": name, "error": "values differ", "max_abs_difference": max_abs})
                    elif not np.array_equal(a, b):
                        sample_report["passed"] = False
                        sample_report["field_differences"].append({"field": name, "error": "string or discrete values differ"})
        sample_reports.append(sample_report)

    report = {
        "reference_root": str(reference_root), "reproduction_root": str(reproduction_root),
        "sample_count": len(sample_reports), "rtol": args.rtol, "atol": args.atol,
        "passed_sample_count": sum(x["passed"] for x in sample_reports),
        "failed_sample_count": sum(not x["passed"] for x in sample_reports),
        "all_pass": all(x["passed"] for x in sample_reports), "samples": sample_reports,
        "comparison_scope": "NPZ field set, dtype, shape, identifiers, metadata, masks, indices, times, and feature arrays. Integer/string/mask arrays are exact; floating arrays use np.allclose with the recorded tolerance.",
    }
    out_dir = reproduction_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reproduction_check.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reference_root / "reports" / "reproduction_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["sample_count", "passed_sample_count", "failed_sample_count", "all_pass", "rtol", "atol"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
