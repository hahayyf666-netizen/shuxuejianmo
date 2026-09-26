"""T3 media-origin validation for Attachment4.

This stage independently locates each Attachment4 clip in its candidate source
media. It never reads model predictions, labels, CSD feature times, or XAI
scores. A successful result is media-navigation evidence only; it does not
claim to recover the official extractor's original window or frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORK = HERE.parents[2]
DEPS = WORK / "q3_native_t0_assets" / "deps"
FFMPEG = WORK / "tools" / "ffmpeg-9.0.2-essentials_build" / "ffmpeg-9.0.2-essentials_build" / "bin" / "ffmpeg.exe"
KNOWN_02 = WORK / "q3_native_t0_assets"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int = 900) -> dict:
    started = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        return {"command": cmd, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
                "elapsed_sec": time.time() - started}
    except subprocess.TimeoutExpired as exc:
        return {"command": cmd, "exit_code": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "TIMEOUT",
                "elapsed_sec": time.time() - started, "timeout_sec": timeout}


def run_yt_dlp(cmd: list[str], timeout: int = 900) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEPS) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return run_cmd([sys.executable, "-m", "yt_dlp", *cmd], env=env, timeout=timeout)


def decode_audio(path: Path) -> np.ndarray:
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
           "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    raw = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def normalized_scan(source: np.ndarray, clip: np.ndarray, sample_rate: int = 16000):
    def scan_once(x: np.ndarray, y0: np.ndarray):
        y = y0 - y0.mean()
        n = len(y)
        nfft = 1 << int((len(x) + n - 1).bit_length())
        corr_full = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(y[::-1], nfft), nfft)
        corr = corr_full[n - 1 : len(x)]
        cs = np.r_[0.0, np.cumsum(x)]
        cs2 = np.r_[0.0, np.cumsum(x * x)]
        energy = (cs2[n:] - cs2[:-n]) - (cs[n:] - cs[:-n]) ** 2 / n
        denom = np.sqrt(np.maximum(energy, 1e-12)) * np.linalg.norm(y)
        corr = corr / denom
        idx = int(np.argmax(corr))
        outside = np.ones(len(corr), dtype=bool)
        outside[max(0, idx - sample_rate) : min(len(corr), idx + sample_rate + 1)] = False
        second = float(np.max(corr[outside])) if outside.any() else -1.0
        return idx, float(corr[idx]), second
    if len(source) < len(clip) or len(clip) < 32:
        raise ValueError("source shorter than clip or clip too short")
    idx, peak, second = scan_once(source, clip)
    half = len(clip) // 2
    first_idx, first_peak, first_second = scan_once(source, clip[:half])
    second_abs_idx, second_peak, second_second = scan_once(source, clip[half:])
    return idx, peak, second, (first_idx, first_peak, first_second), (second_abs_idx, second_peak, second_second), second_abs_idx - half


def decoded_frames(path: Path):
    import av
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * stream.time_base)
            arr = frame.reformat(width=64, height=64, format="gray").to_ndarray().astype(np.float64).reshape(-1)
            arr -= arr.mean()
            norm = np.linalg.norm(arr)
            if norm > 1e-9:
                frames.append((pts, arr / norm))
    return frames


def video_origin(source_video: Path, clip_video: Path, audio_offset: float | None, fractions=(0.2, 0.5, 0.8)) -> dict:
    src = decoded_frames(source_video)
    clip = decoded_frames(clip_video)
    if not src or not clip:
        return {"status": "BLOCKED_DECODE", "source_frame_count": len(src), "clip_frame_count": len(clip)}
    checks = []
    for fraction in fractions:
        requested = fraction * clip[-1][0]
        cp, cvec = min(clip, key=lambda z: abs(z[0] - requested))
        scores = [(sp, float(np.dot(cvec, svec))) for sp, svec in src]
        sp, score = max(scores, key=lambda z: z[1])
        checks.append({"fraction": fraction, "requested_clip_sec": requested, "actual_clip_pts_sec": cp,
                       "matched_source_pts_sec": sp, "offset_sec": sp - cp, "pixel_pearson": score})
    offsets = np.array([x["offset_sec"] for x in checks], dtype=float)
    scores = np.array([x["pixel_pearson"] for x in checks], dtype=float)
    passed = int(np.sum(scores >= 0.55)) >= 2 and float(np.ptp(offsets)) <= 0.12
    return {"status": "PASS" if passed else "FAIL_VISUAL_GATE", "source_frame_count": len(src),
            "clip_frame_count": len(clip), "checks": checks, "offset_median_sec": float(np.median(offsets)),
            "offset_range_sec": float(np.ptp(offsets)), "pearson_min": float(np.min(scores)),
            "pearson_median": float(np.median(scores)), "audio_video_offset_difference_sec":
                None if audio_offset is None else float(np.median(offsets) - audio_offset),
            "thresholds": {"min_two_correlations": 0.55, "max_offset_range_sec": 0.12}}


def clip_audio_gate(source_audio: Path, clip_video: Path) -> dict:
    source = decode_audio(source_audio)
    clip = decode_audio(clip_video)
    idx, peak, second, first, second_half, second_idx = normalized_scan(source, clip)
    out = {"source_audio_duration_sec": len(source) / 16000, "clip_audio_duration_sec": len(clip) / 16000,
           "full_peak_source_offset_sec": idx / 16000, "full_peak_corr": peak,
           "best_other_corr_outside_one_second": second, "peak_margin": peak - second,
           "first_half_peak_corr": first[1], "first_half_offset_sec": first[0] / 16000,
           "second_half_peak_corr": second_half[1], "second_half_offset_sec": second_idx / 16000}
    out["status"] = "PASS" if (peak >= 0.6 and peak - second >= 0.1 and second < 0.5 and
                                 first[1] >= 0.5 and second_half[1] >= 0.5 and
                                 abs(first[0] - idx) / 16000 <= 0.12 and
                                 abs(second_idx - idx) / 16000 <= 0.12) else "FAIL_AUDIO_GATE"
    out["thresholds"] = {"peak_at_least": 0.6, "peak_margin_at_least": 0.1,
                         "best_other_below": 0.5, "half_peak_at_least": 0.5,
                         "half_offset_agreement_sec": 0.12}
    return out


def source_media_for_02() -> tuple[Path, Path]:
    return KNOWN_02 / "YCEllKyaCrc_source_audio.webm", KNOWN_02 / "YCEllKyaCrc_source_video.mp4"


def media_file_for_download(media_dir: Path, sample_id: str, video_id: str) -> Path | None:
    candidates = sorted(media_dir.glob(f"{sample_id}_{video_id}.*"))
    return candidates[0] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--attachment-videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--media-dir", type=Path, required=True)
    ap.add_argument("--max-downloads", type=int, default=15)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.media_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = json.loads(args.candidate_json.read_text(encoding="utf-8"))["samples"]
    candidates = []
    for row in all_candidates:
        top = (row.get("top_candidates") or [{}])[0]
        vid = str(top.get("video_id", ""))
        if len(vid) == 11 and all(ch.isalnum() or ch in "_-" for ch in vid):
            candidates.append(row)
    contract = {"version": "q3-t3-media-origin-v1-2026-09-26",
                "sample_scope": "Attachment4 candidate source IDs with 11-character YouTube shape",
                "excluded_non_youtube_shape_sample_ids": [str(r["sample_id"]) for r in all_candidates if r not in candidates],
                "candidate_selection": "T2 transcript-only top candidate; no model, label or XAI result",
                "audio_rule": {"decode": "first audio stream, mono 16kHz f32le", "whole_source_normalized_correlation": True,
                               "peak_at_least": 0.6, "peak_margin_at_least": 0.1, "best_other_below": 0.5,
                               "half_peak_at_least": 0.5, "half_offset_agreement_sec": 0.12},
                "video_rule": {"decode": "all presentation PTS frames via PyAV, grayscale 64x64",
                               "clip_frame_fractions": [0.2, 0.5, 0.8], "min_two_correlations": 0.55,
                               "max_offset_range_sec": 0.12},
                "identity_rule": "media-origin PASS requires both audio and video gates; partial gates remain non-pass",
                "scope_boundary": "A PASS is clip-origin navigation evidence, not official extractor provenance; no CSD time is used",
                "no_model_or_prediction_run": True}
    (args.out / "t3_frozen_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = []
    commands = []
    per_sample_dir = args.out / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    def save_record(record: dict) -> None:
        (per_sample_dir / f"sample_{record['sample_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for index, row in enumerate(candidates, 1):
        sid = str(row["sample_id"])
        top = (row.get("top_candidates") or [{}])[0]
        vid = str(top.get("video_id", ""))
        rec = {"sample_id": sid, "source_video_id": vid, "candidate": top,
               "source_url": f"https://www.youtube.com/watch?v={vid}", "status": "STARTED"}
        clip = args.attachment_videos / f"{sid}.mp4"
        if not clip.is_file():
            rec["status"] = "BLOCKED_ATTACHMENT_CLIP_MISSING"
            records.append(rec); save_record(rec); continue
        rec["attachment_clip"] = {"path": str(clip), "bytes": clip.stat().st_size, "sha256": sha256_file(clip)}
        if sid == "02" and vid == "YCEllKyaCrc" and (KNOWN_02 / "YCEllKyaCrc_source_audio.webm").is_file():
            audio_path, video_path = source_media_for_02()
            rec["download"] = "REUSED_T1_VERIFIED_MEDIA"
        else:
            probe = run_yt_dlp(["--no-playlist", "--dump-single-json", "--skip-download", rec["source_url"]], timeout=90)
            commands.append({"sample_id": sid, "stage": "metadata_probe", **probe})
            if probe["exit_code"] != 0:
                rec["status"] = "BLOCKED_SOURCE_METADATA"
                rec["probe_error"] = probe["stderr"][-1000:]
                records.append(rec); save_record(rec); print(f"[{index}/{len(candidates)}] {sid} metadata BLOCKED", flush=True); continue
            try:
                meta = json.loads(probe["stdout"])
                rec["metadata"] = {k: meta.get(k) for k in ("id", "title", "duration", "webpage_url", "format_id")}
            except Exception as exc:
                rec["status"] = "BLOCKED_METADATA_PARSE"
                rec["probe_error"] = repr(exc)
                records.append(rec); save_record(rec); continue
            template = args.media_dir / f"{sid}_{vid}.%(ext)s"
            download_cmds = [
                ["--no-playlist", "--format", "best[height<=480][ext=mp4]/best[height<=480]/best",
                 "--merge-output-format", "mp4", "--ffmpeg-location", str(FFMPEG.parent), "--no-part", "--output", str(template), rec["source_url"]],
                ["--no-playlist", "--format", "bestvideo[height<=480]+bestaudio/best",
                 "--merge-output-format", "mp4", "--ffmpeg-location", str(FFMPEG.parent), "--no-part", "--output", str(template), rec["source_url"]],
                ["--no-playlist", "--format", "bestvideo+bestaudio/best",
                 "--merge-output-format", "mp4", "--ffmpeg-location", str(FFMPEG.parent), "--no-part", "--output", str(template), rec["source_url"]],
            ]
            download = None
            for attempt, dcmd in enumerate(download_cmds, 1):
                candidate_download = run_yt_dlp(dcmd, timeout=900)
                commands.append({"sample_id": sid, "stage": "media_download", "attempt": attempt, **candidate_download})
                download = candidate_download
                if candidate_download["exit_code"] == 0:
                    break
            media = media_file_for_download(args.media_dir, sid, vid)
            if download["exit_code"] != 0 or media is None:
                rec["status"] = "BLOCKED_MEDIA_DOWNLOAD"
                rec["download_error"] = download["stderr"][-1500:]
                records.append(rec); save_record(rec); print(f"[{index}/{len(candidates)}] {sid} download BLOCKED", flush=True); continue
            audio_path = media
            video_path = media
            rec["download"] = {"path": str(media), "bytes": media.stat().st_size, "sha256": sha256_file(media),
                                "command_exit_code": download["exit_code"]}
        try:
            audio = clip_audio_gate(audio_path, clip)
            visual = video_origin(video_path, clip, audio.get("full_peak_source_offset_sec"))
            rec["audio"] = audio
            rec["video"] = visual
            rec["media_identity"] = "PASS" if audio["status"] == "PASS" and visual["status"] == "PASS" else "FAIL_OR_PARTIAL"
            rec["local_time_gate"] = "PASS" if rec["media_identity"] == "PASS" else "BLOCKED"
            rec["formal_xai_mapping_status"] = "index_only"
            rec["status"] = "COMPLETE"
        except Exception as exc:
            rec["status"] = "BLOCKED_VALIDATION_ERROR"
            rec["error"] = {"type": type(exc).__name__, "message": str(exc)}
        records.append(rec)
        print(f"[{index}/{len(candidates)}] {sid} {rec['status']} {rec.get('local_time_gate','')}", flush=True)
        save_record(rec)
    aggregate = {"version": contract["version"], "sample_count": len(records),
                 "candidate_count": len(candidates),
                 "source_media_pass_count": sum(r.get("local_time_gate") == "PASS" for r in records),
                 "audio_pass_count": sum(r.get("audio", {}).get("status") == "PASS" for r in records),
                 "video_pass_count": sum(r.get("video", {}).get("status") == "PASS" for r in records),
                 "blocked_or_failed": [r["sample_id"] for r in records if r.get("local_time_gate") != "PASS"],
                 "formal_xai_mapping_status": "index_only",
                 "model_or_prediction_run": False,
                 "records_sha256": None}
    payload = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    (args.out / "t3_media_origin_records.json").write_text(payload, encoding="utf-8")
    aggregate["records_sha256"] = sha256_file(args.out / "t3_media_origin_records.json")
    (args.out / "t3_aggregate_gate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
