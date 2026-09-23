from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "_e_data_audit_out"
OUT.mkdir(exist_ok=True)


def find_root() -> Path:
    c = sorted(p for p in (ROOT / "E题数据").glob("附件3*") if p.is_dir())
    if len(c) != 1:
        raise RuntimeError(f"Expected one attachment3 directory, got {c}")
    return c[0]


def version_of(path: Path) -> str:
    s = path.as_posix().lower()
    if "未对齐" in s or "unaligned" in s:
        return "unaligned"
    if "对齐" in s or "aligned" in s:
        return "aligned"
    return "unknown"


def sample_no(path: Path):
    m = re.search(r"(\d+)(?=\.pkl$)", path.name)
    return int(m.group(1)) if m else None


def unwrap_sample(d):
    if isinstance(d, dict) and set(d.keys()) == {"test"} and isinstance(d["test"], dict):
        return d["test"], True
    return d, False


def arr_meta(x):
    a = np.asarray(x)
    out = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if np.issubdtype(a.dtype, np.number):
        out["nan"] = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
        out["inf"] = int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    return out


def numeric_nonfinite(fields):
    bad = []
    for k, meta in fields.items():
        if meta.get("nan", 0) or meta.get("inf", 0):
            bad.append(k)
    return bad


def main():
    root = find_root()
    files = sorted(root.rglob("*.pkl"))
    by_version = {"aligned": [], "unaligned": [], "unknown": []}
    for p in files:
        by_version[version_of(p)].append(p)

    report = {"attachment": "attachment3", "root": root.as_posix(), "versions": {}}
    hard_failures = []

    expected_ids = set(range(1, 31))
    ids_by_version = {
        ver: [sample_no(p) for p in by_version[ver]]
        for ver in ["aligned", "unaligned"]
    }
    report["sample_ids"] = {
        ver: {
            "ids": sorted(x for x in ids_by_version[ver] if x is not None),
            "missing": sorted(expected_ids - set(x for x in ids_by_version[ver] if x is not None)),
            "duplicates": sorted(k for k, v in Counter(ids_by_version[ver]).items() if k is not None and v > 1),
            "unparseable_count": sum(x is None for x in ids_by_version[ver]),
        }
        for ver in ["aligned", "unaligned"]
    }
    for ver in ["aligned", "unaligned"]:
        m = report["sample_ids"][ver]
        if m["missing"] or m["duplicates"] or m["unparseable_count"]:
            hard_failures.append(f"{ver} sample-number completeness/pairing failed")
    if set(ids_by_version["aligned"]) != set(ids_by_version["unaligned"]):
        hard_failures.append("aligned/unaligned sample-number sets differ")

    for ver in ["aligned", "unaligned"]:
        rows = []
        for p in by_version[ver]:
            with p.open("rb") as f:
                top = pickle.load(f)
            d, wrapped = unwrap_sample(top)
            if not isinstance(d, dict):
                rows.append({"path": p.as_posix(), "error": "sample_not_dict"})
                continue

            item = {
                "path": p.as_posix(),
                "top_level_keys": sorted(map(str, top.keys())) if isinstance(top, dict) else None,
                "wrapped_test_dict": wrapped,
                "keys": sorted(map(str, d.keys())),
                "fields": {},
            }
            for k, v in d.items():
                try:
                    item["fields"][str(k)] = arr_meta(v)
                except Exception:
                    item["fields"][str(k)] = {"type": type(v).__name__}
            item["nonfinite_fields"] = numeric_nonfinite(item["fields"])

            if "text_bert" in d:
                tb = np.asarray(d["text_bert"])
                item["text_bert_checks"] = {
                    "integer_valued": bool(np.all(np.isfinite(tb)) and np.all(tb == np.rint(tb))),
                    "token_min": float(np.nanmin(tb[:, 0, :])) if tb.ndim == 3 and tb.shape[1] >= 1 else None,
                    "token_max": float(np.nanmax(tb[:, 0, :])) if tb.ndim == 3 and tb.shape[1] >= 1 else None,
                    "attention_mask_binary": bool(np.all(np.isin(tb[:, 1, :], [0, 1]))) if tb.ndim == 3 and tb.shape[1] >= 2 else False,
                    "segment_integer_valued": bool(np.all(tb[:, 2, :] == np.rint(tb[:, 2, :]))) if tb.ndim == 3 and tb.shape[1] >= 3 else False,
                }
            rows.append(item)

        expected_n = 30
        if len(rows) != expected_n:
            hard_failures.append(f"{ver}_count={len(rows)} expected={expected_n}")

        keysets = Counter(tuple(x.get("keys", [])) for x in rows)
        ver_report = {
            "count": len(rows),
            "keysets": {str(k): v for k, v in keysets.items()},
            "samples": rows,
        }

        nonfinite_files = [x["path"] for x in rows if x.get("nonfinite_fields")]
        ver_report["nonfinite_files"] = nonfinite_files
        if nonfinite_files:
            hard_failures.append(f"{ver} contains NaN/Inf")

        if ver == "aligned":
            expected = {"text_bert", "audio", "vision"}
            bad_keys = [x["path"] for x in rows if set(x.get("keys", [])) != expected]
            bad_shapes = []
            bad_tb = []
            bad_dtypes = []
            for x in rows:
                fs = x.get("fields", {})
                if (
                    fs.get("text_bert", {}).get("shape") != [1, 3, 50]
                    or fs.get("audio", {}).get("shape") != [1, 50, 74]
                    or fs.get("vision", {}).get("shape") != [1, 50, 35]
                ):
                    bad_shapes.append(x["path"])
                if fs.get("text_bert", {}).get("dtype") != "float32":
                    bad_dtypes.append(x["path"])
                c = x.get("text_bert_checks", {})
                if not (c.get("integer_valued") and c.get("attention_mask_binary") and c.get("segment_integer_valued")):
                    bad_tb.append(x["path"])
            ver_report["unexpected_key_files"] = bad_keys
            ver_report["unexpected_shape_files"] = bad_shapes
            ver_report["unexpected_text_bert_dtype_files"] = bad_dtypes
            ver_report["text_bert_invalid_files"] = bad_tb
            if bad_keys or bad_shapes or bad_tb:
                hard_failures.append("aligned interface/text_bert validation failed")
        else:
            expected = {"raw_text", "audio", "vision"}
            bad_keys = [x["path"] for x in rows if set(x.get("keys", [])) != expected]
            bad_shapes = []
            for x in rows:
                fs = x.get("fields", {})
                if (
                    fs.get("audio", {}).get("shape") != [1, 500, 74]
                    or fs.get("vision", {}).get("shape") != [1, 500, 35]
                ):
                    bad_shapes.append(x["path"])
            ver_report["unexpected_key_files"] = bad_keys
            ver_report["unexpected_shape_files"] = bad_shapes
            ver_report["missing_required_for_attachment2_style_unaligned"] = [
                "text", "text_bert", "id", "audio_lengths", "vision_lengths"
            ]
            if bad_keys or bad_shapes:
                hard_failures.append("unaligned interface validation failed")

        report["versions"][ver] = ver_report

    report["unknown_version_files"] = [p.as_posix() for p in by_version["unknown"]]
    if report["unknown_version_files"]:
        hard_failures.append("unknown-version pkl files exist")

    report["hard_failures"] = hard_failures
    report["status"] = "PASS_WITH_INTERFACE_RISK" if not hard_failures else "FAIL"

    (OUT / "attachment3_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "E题附件3数据审计",
        "=" * 60,
        f"status={report['status']}",
        f"aligned_count={report['versions']['aligned']['count']} unaligned_count={report['versions']['unaligned']['count']}",
        f"sample_ids={report['sample_ids']}",
        f"aligned_text_bert_invalid={len(report['versions']['aligned']['text_bert_invalid_files'])}",
        f"aligned_unexpected_keys={len(report['versions']['aligned']['unexpected_key_files'])}",
        f"aligned_nonfinite={len(report['versions']['aligned']['nonfinite_files'])}",
        f"unaligned_unexpected_keys={len(report['versions']['unaligned']['unexpected_key_files'])}",
        f"unaligned_nonfinite={len(report['versions']['unaligned']['nonfinite_files'])}",
        "unaligned_missing_fields=text,text_bert,id,audio_lengths,vision_lengths",
        f"hard_failures={hard_failures}",
    ]
    (OUT / "attachment3_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
