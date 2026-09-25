"""Verify the published Q1 main figure against the frozen Q1 delivery ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "main_figure_publication_manifest.json").read_text(encoding="utf-8"))
    assert sha256(args.archive) == manifest["source_archive_sha256"]
    for name, record in manifest["published_files"].items():
        assert sha256(here / name) == record["sha256"], name
    with ZipFile(args.archive) as archive:
        for name in (
            "typical_correspondence_s9qJ7ATP7w_clip6.png",
            "typical_correspondence_s9qJ7ATP7w_clip6.json",
            "figure_layout_check.json",
        ):
            assert (here / name).read_bytes() == archive.read(manifest["source_member_directory"] + name)
    svg = (here / "typical_correspondence_s9qJ7ATP7w_clip6_standalone.svg").read_text(encoding="utf-8")
    ET.fromstring(svg)
    assert svg.count("data:image/png;base64,") == 6
    assert "C:\\Users\\Fine" not in svg
    with Image.open(here / "typical_correspondence_s9qJ7ATP7w_clip6.png") as image:
        assert image.size == (3750, 2160)
    evidence = json.loads((here / "typical_correspondence_s9qJ7ATP7w_clip6.json").read_text(encoding="utf-8"))
    assert evidence["sample_key"] == manifest["sample_key"]
    assert len(evidence["word_rows"]) == 5
    assert all(row["audio_indices_match_recomputed_center_rule"] and row["video_indices_match_recomputed_center_rule"] for row in evidence["word_rows"])
    print("PASS: archive/hash integrity, six embedded SVG rasters, five center-rule rows, and PNG geometry")


if __name__ == "__main__":
    main()
