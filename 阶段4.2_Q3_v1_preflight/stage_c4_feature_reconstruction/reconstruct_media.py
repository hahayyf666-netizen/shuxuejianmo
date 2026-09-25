"""Compare raw-MP4-derived acoustic/face candidates to official Attachment 4 rows.

These openSMILE/MediaPipe candidates test reconstructability. Different channel
schemas and an unverified sampling grid make their CKA values diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import traceback

import av
import mediapipe as mp
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return np.divide(values, std, out=np.zeros_like(values), where=std > 1e-12)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float | None:
    """Dimension-free sequence comparison; it cannot prove row identity."""
    if len(x) < 5 or len(x) != len(y):
        return None
    left, right = standardize(x), standardize(y)
    gx, gy = left @ left.T, right @ right.T
    denom = float(np.linalg.norm(gx) * np.linalg.norm(gy))
    return float(np.sum(gx * gy) / denom) if denom > 1e-12 else None


def interpolate_at(centers: np.ndarray, values: np.ndarray,
                   times: np.ndarray) -> tuple[np.ndarray, float]:
    if len(centers) < 2 or np.any(np.diff(centers) <= 0):
        raise ValueError("candidate acoustic frame centers are not strictly increasing")
    sample = np.column_stack([np.interp(times, centers, values[:, col])
                              for col in range(values.shape[1])])
    coverage = float(np.mean((times >= centers[0]) & (times <= centers[-1])))
    return sample.astype(np.float32), coverage


def f0_correlation(official: np.ndarray, candidate: np.ndarray,
                   feature_names: list[str]) -> dict:
    matching = [i for i, name in enumerate(feature_names)
                if "F0semitoneFrom27.5Hz" in name]
    if len(matching) != 1:
        return {"status": "F0 proxy channel absent"}
    semitones = candidate[:, matching[0]].astype(np.float64)
    reconstructed_hz = np.where(semitones > 0, 27.5 * 2 ** (semitones / 12), 0.0)
    first_official_channel = official[:, 0].astype(np.float64)
    both_voiced = (first_official_channel > 0) & (reconstructed_hz > 0)
    if np.count_nonzero(both_voiced) < 8:
        return {"status": "too few jointly positive positions",
                "joint_positive_positions": int(np.count_nonzero(both_voiced))}
    pair = np.corrcoef(first_official_channel[both_voiced], reconstructed_hz[both_voiced])
    return {"status": "hypothetical_F0_channel_comparison_only",
            "joint_positive_positions": int(np.count_nonzero(both_voiced)),
            "pearson": float(pair[0, 1]) if np.isfinite(pair[0, 1]) else None,
            "median_absolute_hz_error": float(np.median(np.abs(
                first_official_channel[both_voiced] - reconstructed_hz[both_voiced])))}


def video_candidate(video: Path, length: int, landmarker) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode real PTS and select nearest decoded frame on a diagnostic 15 Hz grid."""
    rows = []
    chosen_pts = []
    face_valid = []
    target_index = 0
    first_pts = None
    previous = None
    last_detection_pts = None
    last_detection = None
    decoded_count = 0

    def detect(chosen):
        nonlocal last_detection_pts, last_detection
        pts, frame = chosen
        if last_detection_pts == pts:
            return last_detection
        if last_detection_pts is not None and pts < last_detection_pts:
            raise ValueError("chosen video PTS regressed")
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=frame.to_ndarray(format="rgb24"))
        timestamp_ms = int(round((pts - first_pts) * 1000))
        result = landmarker.detect_for_video(image, timestamp_ms)
        sets = result.face_blendshapes or []
        if not sets:
            value = (np.zeros(52, dtype=np.float32), False)
        else:
            categories = {part.category_name: float(part.score) for part in sets[0]}
            if set(categories) != set(core.BLENDSHAPE_NAMES):
                raise ValueError("MediaPipe blendshape schema differs from Q1 asset")
            value = (np.asarray([categories[name] for name in core.BLENDSHAPE_NAMES],
                                dtype=np.float32), True)
        last_detection_pts = pts
        last_detection = value
        return value

    with av.open(str(video), mode="r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("decoded video frame lacks presentation PTS")
            pts = float(frame.pts * frame.time_base)
            if previous is not None and pts < previous[0]:
                raise ValueError("decoded video PTS regressed")
            decoded_count += 1
            if first_pts is None:
                first_pts = pts
            if previous is not None:
                while target_index < length:
                    target = first_pts + (target_index + 0.5) / 15.0
                    if target > pts:
                        break
                    chosen = previous if abs(previous[0] - target) <= abs(pts - target) else (pts, frame)
                    vector, valid = detect(chosen)
                    rows.append(vector)
                    chosen_pts.append(chosen[0])
                    face_valid.append(valid)
                    target_index += 1
            previous = (pts, frame)
        if previous is None:
            raise ValueError("no decoded video frames")
        while target_index < length:
            target = first_pts + (target_index + 0.5) / 15.0
            if target > previous[0] + 1 / 30:
                break
            vector, valid = detect(previous)
            rows.append(vector)
            chosen_pts.append(previous[0])
            face_valid.append(valid)
            target_index += 1
    vectors = np.zeros((length, 52), dtype=np.float32)
    pts_out = np.full(length, np.nan, dtype=np.float64)
    valid_out = np.zeros(length, dtype=np.uint8)
    if rows:
        vectors[:len(rows)] = np.asarray(rows)
        pts_out[:len(rows)] = chosen_pts
        valid_out[:len(rows)] = face_valid
    return vectors, pts_out, valid_out, decoded_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unaligned", type=Path, required=True)
    parser.add_argument("--pair-inventory", type=Path, required=True)
    parser.add_argument("--q1-core-dir", type=Path, required=True)
    parser.add_argument("--face-task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.out / "candidate_npz"
    candidate_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.q1_core_dir.resolve()))
    global core
    import q1_feature_core as core
    with args.pair_inventory.open(newline="", encoding="utf-8-sig") as stream:
        inventory = {row["sample_id"]: row for row in csv.DictReader(stream)}
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.face_task)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    rows = []
    for number in range(1, 21):
        sid = f"{number:02d}"
        pkl = args.unaligned / f"{sid}.pkl"
        video = args.unaligned / "videos" / f"{sid}.mp4"
        with pkl.open("rb") as stream:
            item = pickle.load(stream)
        if str(item.get("id")) != sid:
            raise ValueError(f"{sid}: source identity mismatch")
        if sha256(pkl) != inventory[sid]["unaligned_pkl_sha256"]:
            raise ValueError(f"{sid}: PKL differs from C-2 input")
        if sha256(video) != inventory[sid]["unaligned_video_sha256"]:
            raise ValueError(f"{sid}: MP4 differs from C-2 input")
        audio_len = int(item["audio_lengths"])
        vision_len = int(item["vision_lengths"])
        official_audio = np.asarray(item["audio"][:audio_len], dtype=np.float64)
        official_vision = np.asarray(item["vision"][:vision_len], dtype=np.float64)
        if official_audio.shape != (audio_len, 74) or official_vision.shape != (vision_len, 35):
            raise ValueError(f"{sid}: official source feature shape unexpected")
        record = {"sample_id": sid, "official_audio_rows": audio_len,
                  "official_vision_rows": vision_len,
                  "pkl_sha256": inventory[sid]["unaligned_pkl_sha256"],
                  "video_sha256": inventory[sid]["unaligned_video_sha256"]}
        arrays = {}
        try:
            audio16, audio_info = core.decode_audio_16k(video, 0.0)
            lld = core.opensmile_features(audio16)
            candidate_times = lld["centers"]
            duration = float(audio_info["resampled_duration_sec"])
            grid20 = (np.arange(audio_len, dtype=np.float64) + 0.5) / 20.0
            grid_scaled = (np.arange(audio_len, dtype=np.float64) + 0.5) * duration / audio_len
            proxy20, coverage20 = interpolate_at(candidate_times, lld["values"], grid20)
            proxy_scaled, coverage_scaled = interpolate_at(candidate_times, lld["values"], grid_scaled)
            record["audio"] = {
                "candidate": "openSMILE eGeMAPSv02 LowLevelDescriptors, Q1 validated implementation",
                "candidate_channels": int(proxy20.shape[1]),
                "official_channels": 74,
                "candidate_frame_count": len(candidate_times),
                "decoded_audio_duration_sec": duration,
                "fixed_20hz_grid_coverage": coverage20,
                "duration_scaled_grid_coverage": coverage_scaled,
                "linear_CKA_fixed_20hz": linear_cka(official_audio, proxy20),
                "linear_CKA_duration_scaled": linear_cka(official_audio, proxy_scaled),
                "hypothetical_F0_fixed_20hz": f0_correlation(
                    official_audio, proxy20, lld["feature_names"]),
                "strict_full_74D_verified_rows": 0,
                "failure_reason": "25D eGeMAPS candidate is not a verified 74D official extractor; raw audio time grid is unproven",
                "opensmile_config_sha256": lld["config_sha256"],
            }
            arrays.update(audio_candidate_25D=proxy20, audio_lld_25D=lld["values"],
                          audio_lld_centers_sec=candidate_times)
        except Exception as exc:
            record["audio"] = {"error": str(exc), "traceback": traceback.format_exc(),
                               "strict_full_74D_verified_rows": 0}
        landmarker = None
        try:
            landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            candidate, pts, valid, decoded_count = video_candidate(video, vision_len, landmarker)
            mask = np.isfinite(pts) & (valid > 0)
            record["vision"] = {
                "candidate": "MediaPipe FaceLandmarker 52 blendshapes, Q1 validated model asset",
                "candidate_channels": 52,
                "official_channels": 35,
                "decoded_video_frame_count": decoded_count,
                "proxy_15hz_pts_assigned_rows": int(np.count_nonzero(np.isfinite(pts))),
                "proxy_15hz_face_valid_rows": int(np.count_nonzero(mask)),
                "linear_CKA_on_face_valid_rows": linear_cka(official_vision[mask], candidate[mask]),
                "strict_full_35D_verified_rows": 0,
                "failure_reason": "52D MediaPipe blendshapes do not reproduce the unidentified 35D official visual channels; 15 Hz grid is diagnostic only",
                "mediapipe_task_sha256": sha256(args.face_task),
            }
            arrays.update(vision_candidate_52D=candidate, vision_candidate_pts_sec=pts,
                          vision_candidate_face_valid=valid)
        except Exception as exc:
            record["vision"] = {"error": str(exc), "traceback": traceback.format_exc(),
                                "strict_full_35D_verified_rows": 0}
        finally:
            if landmarker is not None:
                landmarker.close()
        candidate_path = candidate_dir / f"{sid}.npz"
        np.savez_compressed(candidate_path, **arrays)
        record["candidate_npz_sha256"] = sha256(candidate_path)
        rows.append(record)
        print(sid, "audio_CKA", record["audio"].get("linear_CKA_fixed_20hz"),
              "vision_CKA", record["vision"].get("linear_CKA_on_face_valid_rows"),
              "face_valid", record["vision"].get("proxy_15hz_face_valid_rows"), flush=True)
    def finite_values(key: str, field: str) -> list[float]:
        return [float(row[key][field]) for row in rows
                if row[key].get(field) is not None and np.isfinite(row[key][field])]
    audio_cka = finite_values("audio", "linear_CKA_fixed_20hz")
    vision_cka = finite_values("vision", "linear_CKA_on_face_valid_rows")
    report = {
        "purpose": "diagnostic source reconstruction, not feature provenance verification",
        "samples": 20,
        "audio": {
            "official_rows": sum(row["official_audio_rows"] for row in rows),
            "strict_full_row_verified": 0,
            "candidate_schema": "25D eGeMAPSv02 versus 74D official",
            "linear_CKA_fixed_20hz_sample_count": len(audio_cka),
            "linear_CKA_fixed_20hz_median": float(np.median(audio_cka)) if audio_cka else None,
            "samples_with_extraction_error": [row["sample_id"] for row in rows if "error" in row["audio"]],
        },
        "vision": {
            "official_rows": sum(row["official_vision_rows"] for row in rows),
            "strict_full_row_verified": 0,
            "candidate_schema": "52D MediaPipe versus 35D official",
            "linear_CKA_face_valid_sample_count": len(vision_cka),
            "linear_CKA_face_valid_median": float(np.median(vision_cka)) if vision_cka else None,
            "samples_with_extraction_error": [row["sample_id"] for row in rows if "error" in row["vision"]],
        },
        "per_sample": rows,
        "training_run": False,
        "q3_predictor_loaded": False,
    }
    (args.out / "media_reconstruction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
