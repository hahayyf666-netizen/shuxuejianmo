from __future__ import annotations

import hashlib
import re
import unicodedata
import wave
from pathlib import Path
from typing import Any

import av
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from opensmile import FeatureLevel, FeatureSet, Smile

TOL = 1e-6

class HardStop(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def has_letter_or_digit(text: str) -> bool:
    return any(ch.isalnum() or unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def text_stream(text: str) -> str:
    return "".join(ch.casefold() for ch in text if not ch.isspace())


def source_words(text: str) -> list[dict[str, Any]]:
    """Apply the frozen non-whitespace-chunk rule while retaining original spans."""
    words: list[dict[str, Any]] = []
    pending_prefix: int | None = None
    for match in re.finditer(r"\S+", text, flags=re.UNICODE):
        chunk = match.group(0)
        if has_letter_or_digit(chunk):
            start = pending_prefix if pending_prefix is not None else match.start()
            words.append({"char_start": start, "char_end": match.end(), "text": text[start:match.end()]})
            pending_prefix = None
        elif words:
            words[-1]["char_end"] = match.end()
            words[-1]["text"] = text[words[-1]["char_start"]:match.end()]
        else:
            pending_prefix = match.start()
    if pending_prefix is not None:
        raise ValueError("official text ends with punctuation-only content before any word")
    if not words:
        raise ValueError("official transcript contains no output words")
    return words


def camel_name(enum_name: str) -> str:
    parts = enum_name.lower().split("_")
    if parts == ["neutral"]:
        return "_neutral"
    return parts[0] + "".join(part.title() for part in parts[1:])


def decode_audio_16k(path: Path, shared_t0: float):
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise HardStop("no audio stream")
        stream = container.streams.audio[0]
        raw_info = {
            "codec": stream.codec_context.name,
            "time_base": str(stream.time_base),
            "stream_start_time": stream.start_time,
            "stream_start_sec": float(stream.start_time * stream.time_base) if stream.start_time is not None else None,
            "stream_duration_sec": float(stream.duration * stream.time_base) if stream.duration is not None else None,
            "input_sample_rate": stream.codec_context.sample_rate,
            "input_channels": stream.codec_context.channels,
        }
        if raw_info["stream_start_sec"] is None:
            raise HardStop("audio stream start PTS is missing")
        raw_frames = 0
        raw_total = 0
        raw_pts = []
        raw_gaps = []
        previous_end = None
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        output_chunks: list[np.ndarray] = []
        output_pts: list[float] = []
        output_info = []
        def collect(converted):
            if converted is None:
                return
            frames = converted if isinstance(converted, list) else [converted]
            for out in frames:
                if out.pts is None or out.time_base is None:
                    raise HardStop("resampled audio frame is missing presentation PTS")
                pts_sec = float(out.pts * out.time_base)
                arr = np.asarray(out.to_ndarray(), dtype=np.float32).reshape(-1)
                if arr.size != out.samples:
                    raise HardStop("resampled mono frame shape does not match sample count")
                output_pts.append(pts_sec)
                output_chunks.append(arr)
                output_info.append({"pts_sec": pts_sec, "samples": int(out.samples), "rate": int(out.sample_rate)})
        for frame in container.decode(stream):
            raw_frames += 1
            raw_total += int(frame.samples)
            if frame.pts is None or frame.time_base is None:
                raise HardStop("decoded audio frame is missing presentation PTS")
            pts_sec = float(frame.pts * frame.time_base)
            raw_pts.append(pts_sec)
            if previous_end is not None:
                residual = pts_sec - previous_end
                raw_gaps.append(residual)
            previous_end = pts_sec + frame.samples / frame.sample_rate
            collect(resampler.resample(frame))
        collect(resampler.resample(None))
    if not output_chunks:
        raise HardStop("audio resampling produced no samples")
    waveform = np.concatenate(output_chunks).astype(np.float32, copy=False)
    if not np.isfinite(waveform).all():
        raise HardStop("16 kHz audio contains nonfinite samples")
    start_sec = output_pts[0]
    end_sec = start_sec + len(waveform) / 16000.0
    if abs(start_sec - shared_t0) > 1 / 16000:
        raise HardStop(f"audio presentation start {start_sec:.9f} differs from shared t0 {shared_t0:.9f}")
    expected = start_sec
    max_residual = 0.0
    for frame_info in output_info:
        residual = frame_info["pts_sec"] - expected
        max_residual = max(max_residual, abs(residual))
        if abs(residual) > 1e-8:
            raise HardStop(f"resampled audio has a non-contiguous presentation gap/overlap of {residual:.9f}s")
        expected = frame_info["pts_sec"] + frame_info["samples"] / 16000.0
    raw_max_abs_residual = max((abs(g) for g in raw_gaps), default=0.0)
    if raw_max_abs_residual > 1 / raw_info["input_sample_rate"] + 1e-6:
        raise HardStop(f"decoded source audio contains unexplained PTS discontinuity ({raw_max_abs_residual:.9f}s)")
    info = {
        **raw_info,
        "decoded_input_audio_frames": raw_frames,
        "decoded_input_samples": raw_total,
        "decoded_input_duration_sec": raw_total / raw_info["input_sample_rate"],
        "decoded_input_first_pts_sec": raw_pts[0],
        "decoded_input_last_frame_end_sec": previous_end,
        "decoded_input_max_abs_pts_residual_sec": raw_max_abs_residual,
        "resampler": "PyAV AudioResampler(format=fltp, layout=mono, rate=16000)",
        "resampled_output_rate": 16000,
        "resampled_output_samples": int(waveform.size),
        "resampled_start_sec": start_sec,
        "resampled_end_sec": end_sec,
        "resampled_duration_sec": len(waveform) / 16000.0,
        "resampled_frame_count": len(output_info),
        "resampled_max_abs_pts_residual_sec": max_residual,
        "resampled_frame_pts": output_info,
        "audio_time_map": [{"sample_start": 0, "sample_end": int(waveform.size), "presentation_start_sec": start_sec, "presentation_end_sec": end_sec, "rate": 16000}],
    }
    return waveform, info


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    int16 = np.clip(audio, -1.0, 1.0)
    int16 = np.rint(int16 * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def decode_video_blendshapes(path: Path, t0: float, face_landmarker):
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise HardStop("no video stream")
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        info = {
            "codec": stream.codec_context.name,
            "time_base": str(stream.time_base),
            "stream_start_time": stream.start_time,
            "stream_start_sec": float(stream.start_time * stream.time_base) if stream.start_time is not None else None,
            "stream_duration_sec": float(stream.duration * stream.time_base) if stream.duration is not None else None,
            "average_rate_metadata_only": str(stream.average_rate) if stream.average_rate else None,
            "width": width,
            "height": height,
            "decoder": "PyAV/FFmpeg decoded frame presentation PTS",
        }
        if info["stream_start_sec"] is None:
            raise HardStop("video stream start PTS is missing")
        pts: list[float] = []
        durations: list[float | None] = []
        vectors: list[np.ndarray] = []
        valid: list[bool] = []
        per_frame_counts: list[int] = []
        last_ms = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise HardStop("decoded video frame is missing presentation PTS")
            frame_pts = float(frame.pts * frame.time_base)
            relative_ms = int(round((frame_pts - t0) * 1000.0))
            if last_ms is not None and relative_ms <= last_ms:
                raise HardStop(f"MediaPipe VIDEO millisecond timestamps collide or regress at {frame_pts:.9f}s")
            last_ms = relative_ms
            image_np = frame.to_ndarray(format="rgb24")
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            result = face_landmarker.detect_for_video(image, relative_ms)
            pts.append(frame_pts)
            durations.append(float(frame.duration * frame.time_base) if frame.duration else None)
            found = result.face_blendshapes or []
            per_frame_counts.append(len(found))
            if len(found) == 0:
                vectors.append(np.zeros(52, dtype=np.float32))
                valid.append(False)
                continue
            if len(found) != 1:
                raise HardStop(f"MediaPipe num_faces=1 yielded {len(found)} face blendshape sets")
            categories = found[0]
            names = [c.category_name for c in categories]
            if len(names) != 52 or len(set(names)) != 52:
                raise HardStop(f"MediaPipe returned {len(names)} blendshape categories (expected 52 unique)")
            if set(names) != set(BLENDSHAPE_NAMES):
                raise HardStop("MediaPipe blendshape category names differ from its frozen 52-category schema")
            by_name = {c.category_name: float(c.score) for c in categories}
            vector = np.asarray([by_name[name] for name in BLENDSHAPE_NAMES], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise HardStop("MediaPipe returned nonfinite blendshape values")
            vectors.append(vector)
            valid.append(True)
        if not pts:
            raise HardStop("video decode returned no presentation frames")
        if abs(pts[0] - info["stream_start_sec"]) > 1e-6:
            raise HardStop("first decoded video PTS differs from video stream presentation start")
        deltas = np.diff(np.asarray(pts, dtype=np.float64))
        if np.any(deltas <= 0):
            raise HardStop("decoded video presentation PTS is not strictly increasing")
        stream_end = info["stream_start_sec"] + info["stream_duration_sec"] if info["stream_duration_sec"] is not None else None
        decoded_end = pts[-1] + (durations[-1] or (pts[-1] - pts[-2] if len(pts) > 1 else 0.0))
        # The stream/movie duration is metadata, not proof that a decoded frame
        # covers that interval. Keep it as a diagnostic and derive support only
        # from the last decoded frame's duration (or the final PTS step).
        coverage_end = decoded_end
        if pts[0] > t0 + 1 / 90000 or pts[-1] > coverage_end + TOL:
            raise HardStop("video presentation coverage cannot be reconciled with decoded PTS")
        support_start = np.empty(len(pts), dtype=np.float64)
        support_end = np.empty(len(pts), dtype=np.float64)
        support_start[0] = pts[0]
        support_end[-1] = coverage_end
        for i in range(1, len(pts)):
            midpoint = (pts[i - 1] + pts[i]) / 2.0
            support_end[i - 1] = midpoint
            support_start[i] = midpoint
        info.update({
            "frames_decoded": len(pts),
            "first_pts_sec": pts[0],
            "last_pts_sec": pts[-1],
            "median_pts_delta_sec": float(np.median(deltas)) if len(deltas) else None,
            "max_pts_delta_sec": float(np.max(deltas)) if len(deltas) else None,
            "pts_gap_count_over_1_5x_median": int(np.sum(deltas > np.median(deltas) * 1.5)) if len(deltas) else 0,
            "pts_gaps": [{"left_pts_sec": float(pts[i]), "right_pts_sec": float(pts[i + 1]), "delta_sec": float(deltas[i])} for i in range(len(deltas)) if deltas[i] > np.median(deltas) * 1.5],
            "decoded_last_frame_end_sec": decoded_end,
            "actual_video_coverage_start_sec": pts[0],
            "actual_video_coverage_end_sec": coverage_end,
            "stream_or_movie_end_minus_decoded_frame_end_sec": stream_end - decoded_end if stream_end is not None else None,
            "num_faces_per_frame": per_frame_counts,
            "blendshape_category_names_order": BLENDSHAPE_NAMES,
            "blendshape_order_basis": "mediapipe.tasks.python.vision.face_landmarker.Blendshapes enum order",
        })
    return info, np.asarray(pts, dtype=np.float64), support_start, support_end, np.asarray(vectors, dtype=np.float32), np.asarray(valid, dtype=np.uint8)


def extract_text_features(text: str, words: list[dict[str, Any]], tokenizer, model, device: torch.device):
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt", truncation=False)
    ids = encoded["input_ids"]
    offsets = encoded.pop("offset_mapping")[0].cpu().numpy().astype(np.int32)
    limit = int(getattr(model.config, "max_position_embeddings", 512))
    if ids.shape[1] > limit:
        raise ValueError(f"RoBERTa input length {ids.shape[1]} exceeds {limit}; this pilot text needs word-boundary chunking")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
        token_hidden = output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if token_hidden.shape[1] != 768:
        raise HardStop(f"RoBERTa hidden dimension is {token_hidden.shape[1]}, expected 768")
    assignments: list[list[int]] = [[] for _ in words]
    token_word = np.full(len(offsets), -1, dtype=np.int32)
    tokenizer_failures = []
    for ti, (a, b) in enumerate(offsets.tolist()):
        if a == b:
            continue
        source_slice = text[a:b]
        matches = [wi for wi, word in enumerate(words) if word["char_start"] <= a and b <= word["char_end"]]
        if len(matches) == 1:
            wi = matches[0]
            assignments[wi].append(ti)
            token_word[ti] = wi
        elif len(matches) == 0 and source_slice.isspace():
            continue
        else:
            tokenizer_failures.append({"token_index": ti, "offset": [int(a), int(b)], "text": source_slice, "word_matches": matches})
    features = np.zeros((len(words), 768), dtype=np.float32)
    text_valid = np.ones(len(words), dtype=np.uint8)
    word_token_indices = []
    word_token_indptr = [0]
    for wi, indexes in enumerate(assignments):
        word_token_indices.extend(indexes)
        word_token_indptr.append(len(word_token_indices))
        if not indexes:
            text_valid[wi] = 0
            tokenizer_failures.append({"word_index": wi, "word_text": words[wi]["text"], "reason": "no non-special tokenizer subword mapped to source character span"})
        else:
            features[wi] = token_hidden[indexes].mean(axis=0)
    if not np.isfinite(features).all():
        raise HardStop("RoBERTa returned nonfinite pooled word features")
    return {
        "features": features,
        "valid": text_valid,
        "token_offsets": offsets,
        "token_ids": ids[0].cpu().numpy().astype(np.int32),
        "token_word_index": token_word,
        "word_token_indptr": np.asarray(word_token_indptr, dtype=np.int32),
        "word_token_indices": np.asarray(word_token_indices, dtype=np.int32),
        "tokenizer_failures": tokenizer_failures,
        "input_length": int(ids.shape[1]),
    }


def opensmile_features(audio: np.ndarray):
    smile = Smile(feature_set=FeatureSet.eGeMAPSv02, feature_level=FeatureLevel.LowLevelDescriptors, num_workers=1, multiprocessing=False, verbose=False)
    frame = smile.process_signal(audio, sampling_rate=16000)
    if frame.shape[1] != 25 or len(smile.feature_names) != 25:
        raise HardStop(f"openSMILE produced {frame.shape[1]} columns, expected 25 eGeMAPSv02 LLDs")
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["start", "end"]:
        raise HardStop(f"openSMILE returned no interpretable start/end support intervals: {frame.index.names}")
    starts = np.asarray([x.total_seconds() if hasattr(x, "total_seconds") else float(x) for x in frame.index.get_level_values("start")], dtype=np.float64)
    ends = np.asarray([x.total_seconds() if hasattr(x, "total_seconds") else float(x) for x in frame.index.get_level_values("end")], dtype=np.float64)
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if len(starts) != len(values) or np.any(ends <= starts) or np.any(np.diff(starts) < -TOL):
        raise HardStop("openSMILE start/end support intervals are malformed or nonmonotonic")
    return {
        "values": values,
        "starts": starts,
        "ends": ends,
        "centers": (starts + ends) / 2.0,
        "feature_names": list(smile.feature_names),
        "config_path": str(smile.config_path),
        "config_sha256": sha256(Path(smile.config_path)),
        "feature_set": "eGeMAPSv02",
        "feature_level": "LowLevelDescriptors",
        "frame_count": int(len(values)),
    }


def aggregate_by_center(words, starts, ends, centers, values, valid_frames=None, dim=25):
    L = len(words)
    means_stds = np.zeros((L, dim * 2), dtype=np.float32)
    valid = np.zeros(L, dtype=np.uint8)
    indptr = [0]
    all_indices: list[int] = []
    local_issues = []
    for wi, word in enumerate(words):
        a, b = starts[wi], ends[wi]
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            indptr.append(len(all_indices))
            local_issues.append({"word_index": wi, "reason": "invalid or zero-duration word interval"})
            continue
        indexes = np.flatnonzero((centers >= a) & (centers < b))
        if valid_frames is not None:
            indexes = indexes[np.asarray(valid_frames, dtype=bool)[indexes]]
        all_indices.extend(int(i) for i in indexes)
        indptr.append(len(all_indices))
        if len(indexes) == 0:
            local_issues.append({"word_index": wi, "reason": "no valid feature center in half-open word interval"})
            continue
        selected = values[indexes]
        if selected.shape[1] != dim:
            raise HardStop(f"center assignment expected {dim} columns, got {selected.shape[1]}")
        if not np.isfinite(selected).all():
            local_issues.append({"word_index": wi, "reason": "nonfinite raw feature in assigned support", "feature_indices": np.flatnonzero(~np.isfinite(selected)).tolist()[:20]})
            continue
        means_stds[wi, :dim] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        means_stds[wi, dim:] = selected.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
        valid[wi] = 1
    return means_stds, valid, np.asarray(indptr, dtype=np.int32), np.asarray(all_indices, dtype=np.int32), local_issues

BLENDSHAPE_NAMES = [camel_name(member.name) for member in Blendshapes]
if len(BLENDSHAPE_NAMES) != 52 or len(set(BLENDSHAPE_NAMES)) != 52:
    raise HardStop("MediaPipe enum does not define 52 unique blendshape names")
