from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_ROOT = Path(r"D:\Workspace\数学建模\E题数据\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条")
SAMPLE_KEY = "-s9qJ7ATP7w$_$6"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def decode_audio(path: Path) -> tuple[np.ndarray, float, float]:
    chunks: list[np.ndarray] = []
    pts: list[float] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise RuntimeError("source has no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

        def collect(converted) -> None:
            if converted is None:
                return
            for frame in converted if isinstance(converted, list) else [converted]:
                if frame.pts is None or frame.time_base is None:
                    raise RuntimeError("resampled audio frame has no PTS")
                pts.append(float(frame.pts * frame.time_base))
                chunks.append(np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1))

        for frame in container.decode(stream):
            collect(resampler.resample(frame))
        collect(resampler.resample(None))
    if not chunks:
        raise RuntimeError("decoded audio is empty")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    start = pts[0]
    end = start + len(waveform) / 16000.0
    if not np.isfinite(waveform).all():
        raise RuntimeError("decoded audio contains non-finite values")
    return waveform, start, end


def decode_video_frames(path: Path) -> tuple[list[float], dict[int, np.ndarray]]:
    pts: list[float] = []
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError("source has no video stream")
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError(f"decoded video frame {i} has no PTS")
            pts.append(float(frame.pts * frame.time_base))
            frames[i] = frame.to_ndarray(format="rgb24")
    return pts, frames


def csr_row(indptr: np.ndarray, indices: np.ndarray, row: int) -> np.ndarray:
    return indices[int(indptr[row]):int(indptr[row + 1])].astype(np.int64, copy=False)


def redact_face(frame: np.ndarray, landmarks) -> np.ndarray:
    h, w = frame.shape[:2]
    xs = np.asarray([p.x for p in landmarks], dtype=float)
    ys = np.asarray([p.y for p in landmarks], dtype=float)
    if xs.size == 0:
        raise RuntimeError("face landmarks are empty; refusing to render an unredacted face")
    pad_x = max(0.04, (xs.max() - xs.min()) * 0.18)
    pad_y = max(0.04, (ys.max() - ys.min()) * 0.20)
    x0 = max(0, int((xs.min() - pad_x) * w))
    x1 = min(w, int((xs.max() + pad_x) * w) + 1)
    y0 = max(0, int((ys.min() - pad_y) * h))
    y1 = min(h, int((ys.max() + pad_y) * h) + 1)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("invalid face-redaction box")
    image = Image.fromarray(frame)
    crop = image.crop((x0, y0, x1, y1))
    # Downsample to a coarse mosaic before enlarging; the source frame itself is never written.
    small = crop.resize((max(1, crop.width // 12), max(1, crop.height // 12)), Image.Resampling.BILINEAR)
    image.paste(small.resize(crop.size, Image.Resampling.NEAREST), (x0, y0))
    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an auditable example from one existing Q1 feature file.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "q1" / "v1_delivery")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest = read_rows(output_root / "manifest.csv")
    row = next(r for r in manifest if r["sample_key"] == SAMPLE_KEY)
    npz_path = output_root / row["output_file"]
    source = args.input_root / row["source_relpath"]
    if not source.is_file() or sha256(source) != row["source_mp4_sha256"]:
        raise RuntimeError("source MP4 is absent or its SHA-256 differs from the frozen manifest")
    if sha256(npz_path) != row["output_sha256"]:
        raise RuntimeError("feature file hash differs from the frozen manifest")

    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    if str(arrays["alignment_mode"].item()) != "TRI_MODAL_WORD_VALID":
        raise RuntimeError("fixed example is no longer the selected word-level correspondence sample")
    words = arrays["words"].astype(str).tolist()
    starts = arrays["word_start_sec"].astype(float)
    ends = arrays["word_end_sec"].astype(float)
    if len(words) != 5 or not np.all(arrays["word_time_valid"] == 1):
        raise RuntimeError("fixed example no longer has five valid official word intervals")
    audio_indices = [csr_row(arrays["audio_word_feature_indptr"], arrays["audio_word_feature_indices"], i) for i in range(len(words))]
    video_indices = [csr_row(arrays["vision_word_feature_indptr"], arrays["vision_word_feature_indices"], i) for i in range(len(words))]
    audio_centers = arrays["raw_audio_lld_center_sec"].astype(float)
    video_pts_saved = arrays["raw_video_pts_sec"].astype(float)
    # Recompute center assignment independently from the archived indices.
    assignment_checks = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        expected_audio = np.flatnonzero((audio_centers >= s) & (audio_centers < e))
        expected_video = np.flatnonzero((video_pts_saved >= s) & (video_pts_saved < e) & (arrays["raw_video_face_feature_valid"] == 1))
        assignment_checks.append({
            "word_index": i,
            "word": words[i],
            "start_sec": float(s),
            "end_sec": float(e),
            "audio_indices_match_recomputed_center_rule": np.array_equal(audio_indices[i], expected_audio),
            "video_indices_match_recomputed_center_rule": np.array_equal(video_indices[i], expected_video),
            "audio_window_count": int(len(audio_indices[i])),
            "video_frame_count": int(len(video_indices[i])),
        })
    if not all(x["audio_indices_match_recomputed_center_rule"] and x["video_indices_match_recomputed_center_rule"] for x in assignment_checks):
        raise RuntimeError("stored center-to-word indices do not match recomputed half-open interval queries")

    waveform, audio_start, audio_end = decode_audio(source)
    np.testing.assert_allclose([audio_start, audio_end], arrays["audio_presentation_coverage_sec"], rtol=0, atol=1 / 16000)
    pts_decoded, frame_arrays = decode_video_frames(source)
    np.testing.assert_allclose(pts_decoded, video_pts_saved, rtol=0, atol=1e-7)
    selected_frame_ids = []
    for i, ids in enumerate(video_indices):
        if ids.size == 0:
            raise RuntimeError(f"word {words[i]!r} has no selected video frame to demonstrate")
        target = (starts[i] + ends[i]) / 2
        selected_frame_ids.append(int(ids[np.argmin(np.abs(video_pts_saved[ids] - target))]))

    model_path = PROJECT_ROOT / "work" / "model_assets_a0" / "mediapipe" / "face_landmarker.task"
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    redacted = {}
    try:
        for frame_id in selected_frame_ids:
            frame = frame_arrays[frame_id]
            timestamp_ms = int(round((video_pts_saved[frame_id] - float(arrays["shared_t0_sec"].item())) * 1000.0))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(image, timestamp_ms)
            if not result.face_landmarks:
                raise RuntimeError(f"could not redact face at selected video frame {frame_id}")
            redacted[frame_id] = redact_face(frame, result.face_landmarks[0])
    finally:
        landmarker.close()

    out_dir = output_root / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "typical_correspondence_s9qJ7ATP7w_clip6.png"
    fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.6, 0.9, 1.6])
    ax_wave = fig.add_subplot(grid[0])
    ax_map = fig.add_subplot(grid[1], sharex=ax_wave)
    frame_grid = grid[2].subgridspec(1, len(words), wspace=0.04)

    sample_step = max(1, len(waveform) // 12000)
    sample_ids = np.arange(0, len(waveform), sample_step)
    wave_times = audio_start + (sample_ids + 0.5) / 16000.0
    wave_values = waveform[sample_ids]
    ax_wave.plot(wave_times, wave_values, color="#343A40", linewidth=0.45, rasterized=True)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("A  Decoded source audio on the shared presentation timeline", loc="left", fontweight="bold")
    ax_wave.grid(axis="x", color="#D9DEE3", linewidth=0.5)
    for i, (s, e) in enumerate(zip(starts, ends)):
        color = COLORS[i % len(COLORS)]
        ax_wave.axvspan(s, e, color=color, alpha=0.10, linewidth=0)
        ax_wave.axvline(s, color=color, alpha=0.75, linewidth=0.8, linestyle="--")
        ax_wave.axvline(e, color=color, alpha=0.75, linewidth=0.8, linestyle=":")
        ax_wave.text((s + e) / 2, 0.96, words[i], color=color, fontsize=9, ha="center", va="top", transform=ax_wave.get_xaxis_transform())

    duration = float(arrays["original_effective_duration_sec"].item())
    ax_map.set_title("B  Audio-window centers and video-frame PTS selected by each word interval", loc="left", fontsize=10, fontweight="bold")
    ax_map.set_yticks([1, 0], ["Audio LLD centers", "Video frame PTS"])
    ax_map.set_ylim(-0.5, 1.5)
    ax_map.grid(axis="x", color="#D9DEE3", linewidth=0.5)
    for i, (s, e) in enumerate(zip(starts, ends)):
        color = COLORS[i % len(COLORS)]
        ax_map.axvspan(s, e, color=color, alpha=0.08, linewidth=0)
        ax_map.scatter(audio_centers[audio_indices[i]], np.ones(len(audio_indices[i])), s=9, color=color, edgecolors="none", zorder=3)
        ax_map.scatter(video_pts_saved[video_indices[i]], np.zeros(len(video_indices[i])), s=12, color=color, edgecolors="none", zorder=3)
    unassigned_audio = np.ones(len(audio_centers), dtype=bool)
    for ids in audio_indices:
        unassigned_audio[ids] = False
    unassigned_video = np.ones(len(video_pts_saved), dtype=bool)
    for ids in video_indices:
        unassigned_video[ids] = False
    ax_map.scatter(audio_centers[unassigned_audio], np.ones(int(unassigned_audio.sum())), s=5, color="#ADB5BD", alpha=0.5, edgecolors="none", zorder=2)
    ax_map.scatter(video_pts_saved[unassigned_video], np.zeros(int(unassigned_video.sum())), s=7, color="#ADB5BD", alpha=0.55, edgecolors="none", zorder=2)
    ax_map.set_xlabel("Shared presentation time (s)")

    for i, (word, frame_id, ids) in enumerate(zip(words, selected_frame_ids, video_indices)):
        ax = fig.add_subplot(frame_grid[i])
        ax.imshow(redacted[frame_id])
        pts = float(video_pts_saved[frame_id])
        ax.set_title(f"{word}\nframe PTS {pts:.3f} s", fontsize=9, pad=4)
        ax.axis("off")
    fig.suptitle("Typical correspondence example: “But I just kept going”", fontsize=14, fontweight="bold")
    figure_skill_scripts = Path(r"C:\Users\Fine\.codex\skills\math-modeling-main\tools\figure\scripts")
    if figure_skill_scripts.is_dir():
        sys.path.insert(0, str(figure_skill_scripts))
        from visual_qa import audit_layout, print_report, render_preview

        preview_path = out_dir / "typical_correspondence_s9qJ7ATP7w_clip6_preview.png"
        render_preview(fig, str(preview_path), dpi=150)
        layout_issues = audit_layout(fig)
        layout_verdict = print_report(layout_issues)
        if layout_verdict == "FAIL":
            raise RuntimeError(f"figure layout self-check failed: {layout_issues}")
        (out_dir / "figure_layout_check.json").write_text(
            json.dumps({"verdict": layout_verdict, "issues": layout_issues, "preview_file": str(preview_path.relative_to(output_root))}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        layout_verdict = "NOT_RUN"
        layout_issues = []
    fig.savefig(figure_path, dpi=300, facecolor="white")
    svg_path = figure_path.with_suffix(".svg")
    with matplotlib.rc_context({"svg.image_inline": False}):
        fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    evidence = {
        "sample_key": SAMPLE_KEY,
        "official_text": str(arrays["official_text"].item()),
        "sample_mode": str(arrays["alignment_mode"].item()),
        "text_audio_correspondence": str(arrays["text_audio_correspondence"].item()),
        "text_av_time_mapping_status": str(arrays["text_av_time_mapping_status"].item()),
        "human_content_evidence": "Recorded human listening review confirmed hearing “But I just kept going” in both edit-list presentations; the ignored-edit-list version included an extra preceding sentence.",
        "source_relpath": row["source_relpath"],
        "source_mp4_sha256": row["source_mp4_sha256"],
        "feature_file": row["output_file"],
        "feature_file_sha256": row["output_sha256"],
        "alignment_trace_sha256": str(arrays["alignment_trace_sha256"].item()),
        "shared_t0_sec": float(arrays["shared_t0_sec"].item()),
        "audio_decoded_presentation_start_sec": audio_start,
        "audio_decoded_presentation_end_sec": audio_end,
        "audio_after_last_official_word_interval_sec": max(0.0, audio_end - float(ends[-1])),
        "video_frame_count": int(len(pts_decoded)),
        "audio_lld_window_count": int(len(audio_centers)),
        "audio_feature_dim": int(arrays["raw_audio_lld_values"].shape[1]),
        "video_feature_dim": int(arrays["raw_video_blendshape_values"].shape[1]),
        "audio_word_feature_dim": int(arrays["word_audio_feat"].shape[1]),
        "vision_word_feature_dim": int(arrays["word_vision_feat"].shape[1]),
        "waveform_sample_rate_hz": 16000,
        "word_rows": [
            {
                **assignment_checks[i],
                "word_start_sec": float(starts[i]),
                "word_end_sec": float(ends[i]),
                "audio_feature_indices": audio_indices[i].tolist(),
                "audio_feature_centers_sec": audio_centers[audio_indices[i]].tolist(),
                "video_feature_indices": video_indices[i].tolist(),
                "video_pts_sec": video_pts_saved[video_indices[i]].tolist(),
                "illustration_frame_index": selected_frame_ids[i],
                "illustration_frame_pts_sec": float(video_pts_saved[selected_frame_ids[i]]),
                "word_audio_valid": int(arrays["word_audio_valid"][i]),
                "word_vision_valid": int(arrays["word_vision_valid"][i]),
            }
            for i in range(len(words))
        ],
        "source_audio_sha256": sha256(source),
        "feature_archive_sha256": sha256(npz_path),
        "figure_file": str(figure_path.relative_to(output_root)),
        "figure_sha256": sha256(figure_path),
        "figure_svg_file": str(svg_path.relative_to(output_root)),
        "figure_svg_sha256": sha256(svg_path),
        "figure_svg_assets": [
            {"file": str(p.relative_to(output_root)), "sha256": sha256(p)}
            for p in sorted(figure_path.parent.glob(figure_path.stem + "_image*.png"))
        ],
        "figure_layout_verdict": layout_verdict,
        "figure_layout_issues": layout_issues,
        "figure_privacy_note": "Face regions in the displayed source-frame thumbnails were mosaicked for public report distribution; the source MP4 and saved feature archive were not changed.",
    }
    evidence_path = out_dir / "typical_correspondence_s9qJ7ATP7w_clip6.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(figure_path), "figure_sha256": evidence["figure_sha256"], "evidence": str(evidence_path), "words": assignment_checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
