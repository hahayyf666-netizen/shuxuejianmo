"""Verify archive bytes and all file hashes before using a Q3 XAI server bundle."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def verify(path, expected_sha=None):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected_sha and digest != expected_sha:
        raise ValueError("archive SHA-256 mismatch")
    with ZipFile(path) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names) or bundle.testzip() is not None:
            raise ValueError("duplicate names or ZIP CRC failure")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError("unsafe archive path")
        manifest = json.loads(bundle.read("BUNDLE_MANIFEST.json"))
        if set(names) != set(manifest["files"]) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("archive file set mismatch")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError("file hash mismatch: " + name)
    return {"status": "PASS", "archive_sha256": digest, "verified_files": len(manifest["files"]),
            "source_commit": manifest["source_commit"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.expected_sha), indent=2))
