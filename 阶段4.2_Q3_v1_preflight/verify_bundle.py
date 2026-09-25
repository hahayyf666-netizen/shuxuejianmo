"""Verify every bundled file against the embedded SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as bundle:
        names = set(bundle.namelist())
        manifest = json.loads(bundle.read("SHA256SUMS.json"))
        if names != set(manifest) | {"SHA256SUMS.json"}:
            raise ValueError("archive member list differs from manifest")
        for name, expected in manifest.items():
            actual = hashlib.sha256(bundle.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
        source_manifest_bytes = bundle.read("source_manifest.json")
        source_manifest_sha = bundle.read("SOURCE_MANIFEST_SHA256.txt").decode().strip()
        if hashlib.sha256(source_manifest_bytes).hexdigest() != source_manifest_sha:
            raise ValueError("source manifest digest mismatch")
        source_manifest = json.loads(source_manifest_bytes)
        if any(hashlib.sha256(bundle.read(name)).hexdigest() != expected
               for name, expected in source_manifest.items()):
            raise ValueError("source snapshot content mismatch")
        source_commit = bundle.read("SOURCE_COMMIT.txt").decode().strip()
        if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
            raise ValueError("invalid source commit identifier")
    return {"archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified_files": len(manifest), "source_commit": source_commit,
            "source_manifest_sha256": source_manifest_sha}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).with_name("q3_v1_preflight_server_bundle.zip")
    print(json.dumps(verify(target), indent=2))
