"""Fetch only the small timestamped-words CSD from the pinned candidate mirror."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "results" / "t0_frozen_contract.json").read_text(encoding="utf-8"))
TARGET = HERE.parents[2] / "q3_native_t0_assets" / "CMU_MOSEI_TimestampedWords.csd"


def main() -> None:
    url = CONTRACT["candidate_source"]["timestamped_words_url"]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        digest = hashlib.file_digest(TARGET.open("rb"), "sha256").hexdigest()
        print(json.dumps({"path": str(TARGET), "bytes": TARGET.stat().st_size,
                          "sha256": digest, "reused": True}, ensure_ascii=False))
        return
    tmp = TARGET.with_suffix(".csd.partial")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        if not response.headers.get("Content-Length"):
            raise RuntimeError("cannot verify expected candidate file length")
        expected_bytes = int(response.headers["Content-Length"])
        if expected_bytes > 100_000_000:
            raise RuntimeError("timestamped word candidate unexpectedly large")
        with tmp.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
                    digest.update(chunk)
        actual_bytes = tmp.stat().st_size
        if actual_bytes != expected_bytes:
            raise IOError(f"download incomplete: {actual_bytes}/{expected_bytes}")
        metadata = {"url": url, "bytes": actual_bytes, "sha256": digest.hexdigest(),
                    "etag": response.headers.get("ETag") or response.headers.get("X-Linked-Etag")}
    tmp.replace(TARGET)
    (HERE / "results" / "timestamped_words_source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(TARGET), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
