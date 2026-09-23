from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


CHUNK_SIZE = 90 * 1024 * 1024
IO_CHUNK = 64 * 1024 * 1024
LARGE_FILES = {
    "unaligned_50.pkl",
    "aligned_50.pkl",
}


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(IO_CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def split_file(source: Path, staging_root: Path) -> dict:
    relative_source = source.relative_to(source.parents[2])
    relative_source_text = relative_source.as_posix()
    output_dir = staging_root / relative_source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total_size = 0
    parts = []
    part_index = 1
    with source.open("rb") as source_handle:
        while True:
            part_name = f"{source.name}.part{part_index:03d}"
            part_path = output_dir / part_name
            part_size = 0
            part_digest = hashlib.sha256()
            with part_path.open("wb") as part_handle:
                while part_size < CHUNK_SIZE:
                    block = source_handle.read(min(IO_CHUNK, CHUNK_SIZE - part_size))
                    if not block:
                        break
                    part_handle.write(block)
                    part_digest.update(block)
                    digest.update(block)
                    part_size += len(block)
                    total_size += len(block)
            if part_size == 0:
                part_path.unlink()
                break
            parts.append(
                {
                    "path": part_path.relative_to(staging_root).as_posix(),
                    "size": part_size,
                    "sha256": part_digest.hexdigest(),
                }
            )
            print(f"  {relative_source_text}: wrote {part_name} ({part_size:,} bytes)")
            part_index += 1

    return {
        "original_path": relative_source_text,
        "original_size": total_size,
        "original_sha256": digest.hexdigest(),
        "chunk_size": CHUNK_SIZE,
        "parts": parts,
    }


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(source_root: Path, staging_root: Path) -> None:
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if source.name in LARGE_FILES and source.parent.name == "附件2-数据集特征文件":
            manifest_items.append(split_file(source, staging_root))
            continue
        link_or_copy(source, staging_root / source_root.name / relative)

    manifest = {
        "format": "raw-byte-parts-v1",
        "chunk_size": CHUNK_SIZE,
        "source_directory": source_root.name,
        "items": manifest_items,
    }
    manifest_path = staging_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), staging_root / "merge_dataset.py")
    readme = (
        "# E题数据\n\n"
        "两个大型 PKL 文件已按 90 MiB 做原始字节分片，以适配 GitHub 普通 Git 的单文件限制。\n"
        "下载仓库后，在仓库根目录运行 `python merge_dataset.py --manifest dataset_manifest.json`，"
        "脚本会校验 SHA-256 并还原原始文件。\n"
    )
    (staging_root / "DATASET_README.md").write_text(readme, encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


def merge(manifest_path: Path, force: bool) -> None:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["items"]:
        target = root / item["original_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            raise FileExistsError(f"Target exists; use --force to overwrite: {target}")
        digest = hashlib.sha256()
        total_size = 0
        with target.open("wb") as output:
            for part in item["parts"]:
                part_path = root / part["path"]
                with part_path.open("rb") as source:
                    while True:
                        block = source.read(IO_CHUNK)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        total_size += len(block)
        if total_size != item["original_size"] or digest.hexdigest() != item["original_sha256"]:
            target.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch after merge: {target}")
        print(f"Restored {target} ({total_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("source_root", type=Path)
    prepare_parser.add_argument("staging_root", type=Path)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--manifest", type=Path, required=True)
    merge_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_root, args.staging_root)
    else:
        merge(args.manifest, args.force)


if __name__ == "__main__":
    main()
