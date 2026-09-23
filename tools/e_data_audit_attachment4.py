from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "_e_data_audit_out"
OUT.mkdir(exist_ok=True)


def find_root() -> Path:
    c = sorted(p for p in (ROOT / "E题数据").glob("附件4*") if p.is_dir())
    if len(c) != 1:
        raise RuntimeError(f"Expected one attachment4 directory, got {c}")
    return c[0]


def version_of(path: Path) -> str:
    s = path.as_posix().lower()
    if "未对齐" in s or "unaligned" in s:
        return "unaligned"
    if "对齐" in s or "aligned" in s:
        return "aligned"
    return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def arr_meta(x):
    a = np.asarray(x)
    out = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if np.issubdtype(a.dtype, np.number):
        out["nan"] = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
        out["inf"] = int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    return out


def nonzero_span(x):
    a = np.asarray(x)
    z = np.all(a == 0, axis=-1).reshape(-1)
    nz = np.flatnonzero(~z)
    return int(nz[-1] + 1) if len(nz) else 0, int(len(nz))


def numeric_nonfinite(fields):
    return [k for k, meta in fields.items() if meta.get("nan", 0) or meta.get("inf", 0)]


def main():
    root = find_root()
    pkls = sorted(root.rglob("*.pkl"))
    videos = sorted(root.rglob("*.mp4"))
    by_version = {"aligned": [], "unaligned": [], "unknown": []}
    for p in pkls:
        by_version[version_of(p)].append(p)

    video_by_stem = defaultdict(list)
    for p in videos:
        video_by_stem[p.stem].append({"path": p.as_posix(), "sha256": sha256_file(p)})

    report = {
        "attachment": "attachment4",
        "root": root.as_posix(),
        "video_files": len(videos),
        "video_stems": {},
        "versions": {},
    }
    for stem, items in video_by_stem.items():
        report["video_stems"][stem] = {
            "count": len(items),
            "sha256_unique": sorted(set(x["sha256"] for x in items)),
            "paths": [x["path"] for x in items],
        }

    hard_failures = []
    anomalies = []

    for ver in ["aligned", "unaligned"]:
        rows = []
        for p in by_version[ver]:
            with p.open("rb") as f:
                d = pickle.load(f)
            if not isinstance(d, dict):
                rows.append({"path": p.as_posix(), "error": "not_dict"})
                continue

            item = {
                "path": p.as_posix(),
                "stem": p.stem,
                "keys": sorted(map(str, d.keys())),
                "fields": {},
            }
            for k, v in d.items():
                try:
                    item["fields"][str(k)] = arr_meta(v)
                except Exception:
                    item["fields"][str(k)] = {"type": type(v).__name__}
            item["nonfinite_fields"] = numeric_nonfinite(item["fields"])

            if "vision" in d:
                vs, vc = nonzero_span(d["vision"])
                item["vision_nonzero_span"] = vs
                item["vision_nonzero_count"] = vc
                item["vision_all_zero"] = bool(vc == 0)
            if "audio" in d:
                asp, ac = nonzero_span(d["audio"])
                item["audio_nonzero_span"] = asp
                item["audio_nonzero_count"] = ac
            if "vision_lengths" in d:
                item["vision_length_value"] = int(np.asarray(d["vision_lengths"]).reshape(-1)[0])
            if "audio_lengths" in d:
                item["audio_length_value"] = int(np.asarray(d["audio_lengths"]).reshape(-1)[0])
            rows.append(item)

        if len(rows) != 20:
            hard_failures.append(f"{ver}_pkl_count={len(rows)} expected=20")

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
            expected = {"raw_text", "audio", "vision", "id", "text", "text_bert"}
            bad_keys = [x["path"] for x in rows if set(x.get("keys", [])) != expected]
            bad_shapes = []
            bad_dtypes = []
            for x in rows:
                fs = x.get("fields", {})
                if (
                    fs.get("audio", {}).get("shape") != [50, 74]
                    or fs.get("vision", {}).get("shape") != [50, 35]
                    or fs.get("text", {}).get("shape") != [50, 768]
                    or fs.get("text_bert", {}).get("shape") != [3, 50]
                ):
                    bad_shapes.append(x["path"])
                if fs.get("text_bert", {}).get("dtype") != "int64":
                    bad_dtypes.append(x["path"])
                if x.get("vision_all_zero"):
                    anomalies.append({"version": "aligned", "path": x["path"], "type": "vision_all_zero"})
            ver_report["unexpected_key_files"] = bad_keys
            ver_report["unexpected_shape_files"] = bad_shapes
            ver_report["unexpected_text_bert_dtype_files"] = bad_dtypes
            if bad_keys or bad_shapes or bad_dtypes:
                hard_failures.append("aligned field/shape/dtype validation failed")
        else:
            expected = {
                "raw_text", "audio", "vision", "id", "text", "text_bert",
                "audio_lengths", "vision_lengths"
            }
            bad_keys = [x["path"] for x in rows if set(x.get("keys", [])) != expected]
            bad_shapes = []
            length_mismatches = []
            for x in rows:
                fs = x.get("fields", {})
                if (
                    fs.get("audio", {}).get("shape") != [500, 74]
                    or fs.get("vision", {}).get("shape") != [500, 35]
                    or fs.get("text", {}).get("shape") != [50, 768]
                    or fs.get("text_bert", {}).get("shape") != [3, 50]
                ):
                    bad_shapes.append(x["path"])
                if "vision_length_value" in x and x["vision_length_value"] != x.get("vision_nonzero_span"):
                    length_mismatches.append({
                        "path": x["path"],
                        "vision_lengths": x["vision_length_value"],
                        "vision_nonzero_span": x.get("vision_nonzero_span"),
                        "vision_nonzero_count": x.get("vision_nonzero_count"),
                    })
                if "audio_length_value" in x and x["audio_length_value"] != x.get("audio_nonzero_span"):
                    anomalies.append({
                        "version": "unaligned",
                        "path": x["path"],
                        "type": "audio_length_vs_nonzero_span",
                        "audio_lengths": x["audio_length_value"],
                        "audio_nonzero_span": x.get("audio_nonzero_span"),
                    })
            ver_report["unexpected_key_files"] = bad_keys
            ver_report["unexpected_shape_files"] = bad_shapes
            ver_report["vision_length_vs_nonzero_span_mismatches"] = length_mismatches
            anomalies.extend(
                {"version": "unaligned", "type": "vision_length_vs_nonzero_span", **m}
                for m in length_mismatches
            )
            if bad_keys or bad_shapes:
                hard_failures.append("unaligned field/shape validation failed")

        report["versions"][ver] = ver_report

    report["unknown_version_pkls"] = [p.as_posix() for p in by_version["unknown"]]
    if report["unknown_version_pkls"]:
        hard_failures.append("unknown-version pkl files exist")

    unique_video_sha = set()
    for x in report["video_stems"].values():
        unique_video_sha.update(x["sha256_unique"])
    report["unique_video_content_count"] = len(unique_video_sha)
    if len(unique_video_sha) != 20:
        hard_failures.append(f"unique_video_content_count={len(unique_video_sha)} expected=20")

    report["anomalies"] = anomalies
    report["hard_failures"] = hard_failures
    report["status"] = "PASS_WITH_KNOWN_ANOMALIES" if not hard_failures else "FAIL"

    (OUT / "attachment4_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "E题附件4数据审计",
        "=" * 60,
        f"status={report['status']}",
        f"aligned_count={report['versions']['aligned']['count']} unaligned_count={report['versions']['unaligned']['count']}",
        f"video_files={len(videos)} unique_video_content_count={report['unique_video_content_count']}",
        f"aligned_nonfinite={len(report['versions']['aligned']['nonfinite_files'])}",
        f"unaligned_nonfinite={len(report['versions']['unaligned']['nonfinite_files'])}",
        f"aligned_vision_all_zero={sum(1 for x in anomalies if x.get('type') == 'vision_all_zero')}",
        f"unaligned_vision_length_mismatches={len(report['versions']['unaligned'].get('vision_length_vs_nonzero_span_mismatches', []))}",
        f"hard_failures={hard_failures}",
    ]
    (OUT / "attachment4_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
