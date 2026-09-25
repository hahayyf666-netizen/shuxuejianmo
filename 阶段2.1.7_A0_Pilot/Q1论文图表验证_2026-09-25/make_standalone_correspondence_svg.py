"""Embed the six raster layers of the archived Q1 correspondence SVG.

The original Matplotlib SVG refers to absolute paths on the generating machine.
This transport-only conversion keeps the vectors and raster pixels unchanged.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


PREFIX = "outputs/q1/v1_delivery/examples/"
STEM = "typical_correspondence_s9qJ7ATP7w_clip6"
ORIGINAL_SVG_SHA256 = "cbd2b54c82db49a941a03702443a54d50db22152aba4d5edba108f2c2ac1a596"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Q1 v1 candidate ZIP")
    parser.add_argument("output", type=Path, help="Standalone SVG output")
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        raw = archive.read(f"{PREFIX}{STEM}.svg")
        if sha256(raw) != ORIGINAL_SVG_SHA256:
            raise ValueError("Original SVG SHA-256 differs from frozen package")
        svg = raw.decode("utf-8")
        seen: set[int] = set()

        def embed(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if index in seen:
                raise ValueError(f"Repeated raster layer: {index}")
            seen.add(index)
            raster = archive.read(f"{PREFIX}{STEM}.svg.image{index}.png")
            encoded = base64.b64encode(raster).decode("ascii")
            return f'xlink:href="data:image/png;base64,{encoded}"'

        svg = re.sub(
            rf'xlink:href="[^\"]*{re.escape(STEM)}\.svg\.image([0-5])\.png"',
            embed,
            svg,
        )
    if seen != set(range(6)):
        raise ValueError(f"Expected six embedded layers; found {sorted(seen)}")
    ET.fromstring(svg)
    output = svg.encode("utf-8")
    args.output.write_bytes(output)
    print(f"embedded_rasters=6 svg_sha256={sha256(output)} bytes={len(output)}")


if __name__ == "__main__":
    main()
