"""Recreate the frozen Attachment4 CSV from the frozen JSON, without inference."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from export_predictions_csv import export


EXPECTED_SHA256 = "e219309f0c8460ef34e2ae597e421df00ab9579220b8fa5066422a78acea414c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Frozen attachment4_predictions_explanations.json")
    parser.add_argument("--out", type=Path, required=True, help="New output path; an existing file is never overwritten")
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"frozen JSON not found: {args.source}")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".attachment4-export-", suffix=".csv", dir=args.out.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        export(args.source, temp)
        actual = sha256(temp)
        if actual != EXPECTED_SHA256:
            raise SystemExit(f"STOP: deterministic export hash mismatch; expected={EXPECTED_SHA256}, actual={actual}")
        os.replace(temp, args.out)
        print(f"status=BYTE_IDENTICAL_TO_FROZEN_CSV\nrows=20\nsha256={actual}\nout={args.out}")
    finally:
        if temp.exists():
            temp.unlink()


if __name__ == "__main__":
    main()

