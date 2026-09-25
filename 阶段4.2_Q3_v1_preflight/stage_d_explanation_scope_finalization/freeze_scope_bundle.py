"""Freeze the final scope + server preflight/training entry from committed Git bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
INCLUDE = (
    "requirements-preflight.txt", "frozen_config.json", "verify_bundle.py",
    "run_preflight.py", "run_server_preflight.py", "run_formal_training.py",
    "q3v1/__init__.py", "q3v1/data.py", "q3v1/model.py", "q3v1/explain.py",
    "q3v1/mapping.py", "q3v1/scope_gate.py", "q3v1/train_eval.py", "tests/test_preflight.py",
    "stage_c4_feature_reconstruction/results/c4_gate.json",
    "stage_c4_feature_reconstruction/results/text_row_trace_564.csv",
    "stage_d_explanation_scope_finalization/explanation_scope_contract.json",
    "stage_d_explanation_scope_finalization/Q3_EXPLANATION_SCOPE_FINAL_REPORT.md",
    "stage_d_explanation_scope_finalization/SERVER_HANDOFF.md",
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def main() -> None:
    commit = git("rev-parse", "HEAD").decode().strip()
    source = {}
    for relative in INCLUDE:
        source[relative] = git("show", f"{commit}:{ROOT.name}/{relative}")
    source_manifest = {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(source.items())}
    source_manifest_bytes = (json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n").encode()
    metadata = {
        "SOURCE_COMMIT.txt": (commit + "\n").encode(),
        "source_manifest.json": source_manifest_bytes,
        "SOURCE_MANIFEST_SHA256.txt": (hashlib.sha256(source_manifest_bytes).hexdigest() + "\n").encode(),
    }
    payloads = {**source, **metadata}
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())}
    checksums_bytes = (json.dumps(checksums, indent=2, ensure_ascii=False) + "\n").encode()
    archive = Path(__file__).with_name("q3_scope_preflight_server_bundle.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted({**payloads, "SHA256SUMS.json": checksums_bytes}.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            bundle.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    report = {"source_commit": commit, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
              "bytes": archive.stat().st_size, "files": len(checksums) + 1,
              "training_performed": False, "server_preflight_performed": False}
    archive.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
