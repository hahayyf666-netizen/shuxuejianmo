"""Deterministic server bundle from an already committed source snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
INCLUDE = (
    ".gitignore", ".gitattributes", "使用指南.md", "README.md",
    "requirements-preflight.txt", "frozen_config.json",
    "run_preflight.py", "run_mapping_precheck.py", "freeze_bundle.py", "verify_bundle.py", "q3v1/__init__.py",
    "q3v1/data.py", "q3v1/model.py", "q3v1/explain.py", "q3v1/mapping.py", "q3v1/train_eval.py",
    "tests/test_preflight.py", "reports/preflight_machine_report.json",
    "reports/mapping_precheck.json", "reports/representative_evidence_mapping.csv",
    "reports/valid_xai_subset_120.txt", "reports/train_scaler.npz",
    "reports/unit_tests.stdout.log", "reports/unit_tests.stderr.log",
    "reports/unit_tests.exit_code.txt", "reports/environment_pip_freeze.txt", "preflight_report.md",
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def main() -> None:
    commit = git("rev-parse", "HEAD").decode().strip()
    source_bytes = {}
    for relative in INCLUDE:
        repo_path = ROOT.name + "/" + relative
        try:
            source_bytes[relative] = git("show", f"{commit}:{repo_path}")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"commit source file before freezing: {repo_path}") from error
    source_manifest = {name: hashlib.sha256(payload).hexdigest()
                       for name, payload in sorted(source_bytes.items())}
    source_manifest_bytes = (json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    source_manifest_sha = hashlib.sha256(source_manifest_bytes).hexdigest()
    metadata = {
        "SOURCE_COMMIT.txt": (commit + "\n").encode(),
        "source_manifest.json": source_manifest_bytes,
        "SOURCE_MANIFEST_SHA256.txt": (source_manifest_sha + "\n").encode(),
    }
    for name, payload in metadata.items():
        (ROOT / name).write_bytes(payload)
    payloads = {**source_bytes, **metadata}
    manifest = {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(payloads.items())}
    manifest_path = ROOT / "SHA256SUMS.json"
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    archive = ROOT / "q3_v1_preflight_server_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, payload in sorted({**payloads, "SHA256SUMS.json": manifest_bytes}.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            bundle.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(json.dumps({"archive": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "bytes": archive.stat().st_size, "files": len(manifest) + 1,
                      "source_commit": commit, "source_manifest_sha256": source_manifest_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
