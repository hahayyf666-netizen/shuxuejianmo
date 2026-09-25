"""Summarize raw MP4 decode PTS without assigning feature-row timestamps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_json(executable: Path, args: list[str]) -> tuple[dict, str]:
    command = [str(executable), "-v", "error", *args, "-of", "json"]
    completed = subprocess.run(command, capture_output=True, timeout=45, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe exit={completed.returncode}: {completed.stderr.decode(errors='replace')}")
    return json.loads(completed.stdout), hashlib.sha256(completed.stdout).hexdigest()


def summarize_pts(executable: Path, selector: str, video: Path) -> dict:
    data, stdout_sha = ffprobe_json(executable, ["-select_streams", selector,
                                     "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                                     str(video)])
    pts = [float(frame["best_effort_timestamp_time"]) for frame in data.get("frames", [])
           if "best_effort_timestamp_time" in frame]
    return {"decoded_frames_with_pts": len(pts),
            "first_pts_sec": pts[0] if pts else None,
            "last_pts_sec": pts[-1] if pts else None,
            "pts_finite": all(math.isfinite(value) for value in pts),
            "pts_nondecreasing": all(a <= b for a, b in zip(pts, pts[1:])),
            "ffprobe_stdout_sha256": stdout_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--video-directory", type=Path, required=True)
    parser.add_argument("--pair-inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.pair_inventory.open(encoding="utf-8-sig", newline="") as stream:
        pairs = {row["sample_id"]: row for row in csv.DictReader(stream)}
    version = subprocess.run([str(args.ffprobe), "-version"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.splitlines()[0]
    rows = []
    for number in range(1, 21):
        sample_id = f"{number:02d}"
        video = args.video_directory / f"{sample_id}.mp4"
        media_sha = sha256(video)
        if media_sha != pairs[sample_id]["aligned_video_sha256"]:
            raise ValueError(f"{sample_id}: video SHA differs from pair inventory")
        metadata, metadata_sha = ffprobe_json(args.ffprobe, ["-show_entries",
            "stream=index,codec_type,avg_frame_rate,nb_frames,time_base,start_time,duration:format=duration",
            str(video)])
        streams = {s["codec_type"]: s for s in metadata.get("streams", [])}
        if "video" not in streams or "audio" not in streams:
            raise ValueError(f"{sample_id}: missing original MP4 video or audio stream")
        duration = float(metadata["format"]["duration"])
        video_pts = summarize_pts(args.ffprobe, "v:0", video)
        audio_pts = summarize_pts(args.ffprobe, "a:0", video)
        row = {"sample_id": sample_id, "video_sha256": media_sha,
               "format_duration_sec": duration,
               "video_header_nb_frames": streams["video"].get("nb_frames", ""),
               "video_rate_reported": streams["video"].get("avg_frame_rate", ""),
               "video_decoded_frames_with_pts": video_pts["decoded_frames_with_pts"],
               "video_first_pts_sec": video_pts["first_pts_sec"],
               "video_last_pts_sec": video_pts["last_pts_sec"],
               "video_pts_finite": video_pts["pts_finite"],
               "video_pts_nondecreasing": video_pts["pts_nondecreasing"],
               "audio_decoded_frames_with_pts": audio_pts["decoded_frames_with_pts"],
               "audio_first_pts_sec": audio_pts["first_pts_sec"],
               "audio_last_pts_sec": audio_pts["last_pts_sec"],
               "audio_pts_finite": audio_pts["pts_finite"],
               "audio_pts_nondecreasing": audio_pts["pts_nondecreasing"],
               "unaligned_audio_length": int(pairs[sample_id]["audio_unaligned_length"]),
               "unaligned_vision_length": int(pairs[sample_id]["vision_unaligned_length"]),
               "audio_rows_per_second_proxy_only": int(pairs[sample_id]["audio_unaligned_length"]) / duration,
               "vision_rows_per_second_proxy_only": int(pairs[sample_id]["vision_unaligned_length"]) / duration,
               "audio_feature_row_to_raw_pts_verified": False,
               "vision_feature_row_to_raw_pts_verified": False,
               "metadata_stdout_sha256": metadata_sha,
               "video_pts_stdout_sha256": video_pts["ffprobe_stdout_sha256"],
               "audio_pts_stdout_sha256": audio_pts["ffprobe_stdout_sha256"]}
        rows.append(row)
    with (args.out / "media_pts_20.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"samples": len(rows), "ffprobe_version": version,
               "ffprobe_sha256": sha256(args.ffprobe),
               "pair_inventory_sha256": sha256(args.pair_inventory),
               "samples_with_both_streams_and_pts": sum(
                   r["audio_decoded_frames_with_pts"] > 0 and r["video_decoded_frames_with_pts"] > 0
                   for r in rows),
               "samples_with_finite_nondecreasing_audio_pts": sum(
                   r["audio_pts_finite"] and r["audio_pts_nondecreasing"] for r in rows),
               "samples_with_finite_nondecreasing_video_pts": sum(
                   r["video_pts_finite"] and r["video_pts_nondecreasing"] for r in rows),
               "samples_with_video_header_count_different_from_decoded_pts_count": sum(
                   bool(r["video_header_nb_frames"]) and
                   int(r["video_header_nb_frames"]) != r["video_decoded_frames_with_pts"]
                   for r in rows),
               "audio_feature_row_to_raw_pts_verified": 0,
               "vision_feature_row_to_raw_pts_verified": 0,
               "command_templates": [
                   "ffprobe -v error -show_entries stream=index,codec_type,avg_frame_rate,nb_frames,time_base,start_time,duration:format=duration -of json <MP4>",
                   "ffprobe -v error -select_streams v:0 -show_frames -show_entries frame=best_effort_timestamp_time -of json <MP4>",
                   "ffprobe -v error -select_streams a:0 -show_frames -show_entries frame=best_effort_timestamp_time -of json <MP4>"],
               "note": "Decoded MP4 PTS are verified as raw-media metadata only; no official feature-row time or frame attribution is asserted."}
    (args.out / "media_pts_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
