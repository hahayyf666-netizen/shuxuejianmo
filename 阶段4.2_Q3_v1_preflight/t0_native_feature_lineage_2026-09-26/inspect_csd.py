"""Inspect a pinned CSD candidate without loading its full data file."""

from __future__ import annotations

import argparse
import json

import h5py

from remote_h5 import HTTPRangeFile


def describe(group: h5py.Group | h5py.File, depth: int = 0) -> dict:
    result: dict = {"name": group.name, "children": []}
    for key in list(group.keys())[:8]:
        item = group[key]
        row = {"key": key, "type": type(item).__name__}
        if isinstance(item, h5py.Dataset):
            row.update(shape=item.shape, dtype=str(item.dtype))
        elif depth < 2:
            row["sample"] = describe(item, depth + 1)
        result["children"].append(row)
    result["child_count"] = len(group)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    source = HTTPRangeFile(args.url)
    try:
        with h5py.File(source, "r") as csd:
            summary = describe(csd)
        summary.update(remote_size=source.size, etag=source.etag,
                       requests=source.requests_made, bytes_transferred=source.bytes_transferred)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    finally:
        source.close()


if __name__ == "__main__":
    main()
