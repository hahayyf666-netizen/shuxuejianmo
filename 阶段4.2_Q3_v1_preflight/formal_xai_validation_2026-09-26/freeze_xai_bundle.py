"""Build a reproducible archive from committed, byte-verified Q3 sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
REPO = STAGE.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    own_files = ["run_formal_xai_validation.py", "xai_checks.py", "validation_contract.json", "PROTOCOL_SOURCE.md",
                 "test_xai_checks.py", "verify_xai_bundle.py", "README_SERVER.md"]
    sources = {n: HERE / n for n in own_files}
    sources.update({"assets/" + p.name: p for p in (HERE / "assets").glob("*") if p.is_file()})
    sources.update({"q3v1/" + p.name: p for p in (STAGE / "q3v1").glob("*.py")})
    sources["run_formal_training.py"] = STAGE / "run_formal_training.py"
    payloads = {}
    for name, path in sorted(sources.items()):
        relative = path.relative_to(REPO).as_posix()
        committed = subprocess.check_output(["git", "-C", str(REPO), "show", f"{source_commit}:{relative}"])
        if committed != path.read_bytes():
            raise ValueError("uncommitted or newline-altered source: " + relative)
        payloads[name] = committed
    manifest = {"source_commit": source_commit, "stage": "FORMAL_XAI_SERVER_PREPARATION",
                "files": {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items()}}
    payloads["BUNDLE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "q3_xai_validation_bundle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, data in sorted(payloads.items()):
            info = ZipInfo(name, date_time=(2026, 9, 26, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, data)
    receipt = {"source_commit": source_commit, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "bytes": archive.stat().st_size, "file_count": len(payloads),
               "formal_xai_run": False, "test_evaluated": False, "attachment4_inference": False}
    (args.out / "q3_xai_validation_bundle.manifest.json").write_bytes((json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
