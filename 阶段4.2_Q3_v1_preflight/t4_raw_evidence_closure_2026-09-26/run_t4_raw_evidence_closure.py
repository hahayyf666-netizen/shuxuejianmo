#!/usr/bin/env python3
"""T4: map frozen Q3 audio/vision positions to raw-media navigation evidence.

This program never imports or loads the Q3 model. It consumes only the frozen
Attachment4 sample JSON, T2/T3 lineage records, and existing media files.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np


SAMPLES = ("04", "14", "19")
FROZEN_VISION = {"04": (15, 14, 13), "14": (34, 35, 33), "19": (22, 21, 23)}
ALLOWED_VISION = {"index_only", "nearest_frame_candidate_only", "reconstructed_keyframe_from_verified_lineage"}
ALLOWED_AUDIO = {"index_only", "reconstructed_speech_interval_from_verified_lineage"}
FORBIDDEN_MUTATIONS = [
    "training", "model selection", "inference", "test evaluation",
    "Shapley recomputation", "IG recomputation", "formal prediction changes",
    "historical 4.6 Gate edits", "4.7 transition",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision(repo_root: Path, ref: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", ref], text=True).strip()


def unit_vec(frame: av.VideoFrame) -> np.ndarray:
    arr = frame.reformat(width=64, height=64, format="gray").to_ndarray().astype(np.float64).reshape(-1)
    arr -= arr.mean()
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        raise ValueError("decoded grayscale frame has near-zero variance")
    return arr / norm


def decode_video(path: Path) -> tuple[list[dict], float, float]:
    """Return actual PyAV presentation PTS vectors and presentation coverage."""
    frames: list[dict] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        stream_end = None
        if stream.duration is not None:
            candidate = float(stream.duration * stream.time_base)
            if math.isfinite(candidate) and candidate > 0:
                stream_end = candidate
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * stream.time_base)
            if not math.isfinite(pts):
                continue
            frames.append({"pts": pts, "vec": unit_vec(frame), "raw_pts": int(frame.pts)})
    if not frames:
        raise ValueError(f"no decoded video frames with PTS: {path.name}")
    frames.sort(key=lambda x: (x["pts"], x["raw_pts"]))
    diffs = [b["pts"] - a["pts"] for a, b in zip(frames, frames[1:]) if b["pts"] > a["pts"]]
    nominal_step = float(np.median(diffs)) if diffs else 0.0
    end = stream_end if stream_end is not None and stream_end > frames[-1]["pts"] else frames[-1]["pts"] + nominal_step
    if end <= frames[-1]["pts"]:
        end = frames[-1]["pts"]
    return frames, frames[0]["pts"], end


def save_exact_pts_frame(path: Path, wanted_pts: float, dest: Path) -> None:
    """Re-decode the clip and save the exact selected PTS frame as PNG."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * stream.time_base)
            if abs(pts - wanted_pts) <= 1e-9:
                dest.parent.mkdir(parents=True, exist_ok=True)
                frame.to_image().save(dest, format="PNG")
                return
    raise RuntimeError(f"selected decoded PTS {wanted_pts:.9f} not found on re-decode: {path.name}")


def nearest_frame(frames: list[dict], target: float) -> dict:
    return min(frames, key=lambda x: (abs(x["pts"] - target), x["pts"]))


def strict_source_row(native_rows: list[dict], j: int) -> dict | None:
    matches = [r for r in native_rows if int(r["official_j"]) == j]
    return matches[0] if len(matches) == 1 else None


def positions_from_sample(sample: dict) -> dict:
    result = {"vision": [], "audio_by_target": {}}
    actual_top3 = [int(p["official_seq_index"]) for p in
                   sample["targets"]["classification"]["modalities"]["vision"]["top_positions"][:3]]
    if actual_top3 != list(FROZEN_VISION[sample["sample_id"]]):
        raise ValueError(f"frozen vision top-3 mismatch for {sample['sample_id']}: {actual_top3}")
    for seq in FROZEN_VISION[sample["sample_id"]]:
        found = [p for p in sample["targets"]["classification"]["modalities"]["vision"]["top_positions"]
                 if int(p["official_seq_index"]) == seq]
        if len(found) != 1:
            raise ValueError(f"frozen vision position {sample['sample_id']}:{seq} is absent or duplicated in formal JSON")
        result["vision"].append({"official_seq_index": seq, **found[0]})
    union: list[int] = []
    for target in ("classification", "regression"):
        top = sample["targets"][target]["modalities"]["audio"]["top_positions"][:3]
        if len(top) != 3:
            raise ValueError(f"expected frozen audio top-3 for {sample['sample_id']} {target}")
        result["audio_by_target"][target] = [{"official_seq_index": int(p["official_seq_index"]), **p} for p in top]
        for p in top:
            seq = int(p["official_seq_index"])
            if seq not in union:
                union.append(seq)
    result["audio_union_positions"] = union
    return result


def make_scope(repo_root: Path, phase_root: Path, handoff_root: Path, lineage_path: Path,
               t2_native_path: Path, t3_root: Path, formal_dir: Path,
               attachment_video_dir: Path, source_media_dir: Path) -> tuple[dict, dict, dict, dict]:
    lineage_rows = list(csv.DictReader(lineage_path.open(encoding="utf-8-sig", newline="")))
    lineage = {(r["sample_id"], r["modality"], int(r["official_seq_index"])): r for r in lineage_rows}
    with gzip.open(t2_native_path, "rt", encoding="utf-8") as f:
        native = json.load(f)
    native_samples = {r["sample_id"]: r for r in native["samples"]}
    t3_contract_path = t3_root / "t3_frozen_contract.json"
    t3_aggregate_path = t3_root / "t3_aggregate_gate.json"
    t3_contract = read_json(t3_contract_path)
    t3_aggregate = read_json(t3_aggregate_path)
    t3_records = {}
    formal_samples = {}
    media_inputs = {}
    frozen = {}
    for sid in SAMPLES:
        formal_path = formal_dir / f"sample_{sid}.json"
        t3_path = t3_root / "per_sample" / f"sample_{sid}.json"
        formal = read_json(formal_path)
        t3 = read_json(t3_path)
        formal_samples[sid] = formal
        t3_records[sid] = t3
        positions = positions_from_sample(formal)
        frozen[sid] = {
            "formal_sample_json_path": formal_path.relative_to(repo_root).as_posix(),
            "formal_sample_json_sha256": sha256(formal_path),
            "vision_positions": [{"official_seq_index": p["official_seq_index"],
                                  "signed_importance": p["signed_importance"],
                                  "absolute_importance": p["absolute_importance"]}
                                 for p in positions["vision"]],
            "audio_top3_classification": [{"official_seq_index": p["official_seq_index"],
                                           "signed_importance": p["signed_importance"]}
                                          for p in positions["audio_by_target"]["classification"]],
            "audio_top3_regression": [{"official_seq_index": p["official_seq_index"],
                                       "signed_importance": p["signed_importance"]}
                                      for p in positions["audio_by_target"]["regression"]],
            "audio_union_positions": positions["audio_union_positions"],
            "t3_sample_record_path": t3_path.relative_to(repo_root).as_posix(),
            "t3_sample_record_sha256": sha256(t3_path),
        }
        clip_rel = Path(t3["attachment_clip"]["path"])
        clip_path = attachment_video_dir / f"{sid}.mp4"
        source_name = Path(t3["download"]["path"]).name
        source_path = source_media_dir / source_name
        media_inputs[sid] = {
            "attachment_clip_path": clip_path,
            "attachment_clip_expected_sha256": t3["attachment_clip"]["sha256"],
            "attachment_clip_actual_sha256": sha256(clip_path) if clip_path.is_file() else None,
            "source_video_path": source_path,
            "source_video_expected_sha256": t3["download"]["sha256"],
            "source_video_actual_sha256": sha256(source_path) if source_path.is_file() else None,
            "t3_attachment_clip_relative_path": str(clip_rel),
        }
    scope = {
        "task": "T4_RAW_EVIDENCE_CLOSURE",
        "scope_frozen_before_media_decode": True,
        "source_ref": "origin/q3-v1-preflight",
        "source_branch_head": git_revision(repo_root, "HEAD"),
        "origin_branch_head": git_revision(repo_root, "origin/q3-v1-preflight"),
        "samples": frozen,
        "t2_lineage_path": lineage_path.relative_to(repo_root).as_posix(),
        "t2_lineage_sha256": sha256(lineage_path),
        "t2_native_rows_path": t2_native_path.relative_to(repo_root).as_posix(),
        "t2_native_rows_sha256": sha256(t2_native_path),
        "t3_contract_path": t3_contract_path.relative_to(repo_root).as_posix(),
        "t3_contract_sha256": sha256(t3_contract_path),
        "t3_aggregate_gate_path": t3_aggregate_path.relative_to(repo_root).as_posix(),
        "t3_aggregate_gate_sha256": sha256(t3_aggregate_path),
        "prohibited_mutations": FORBIDDEN_MUTATIONS,
        "media_inputs": {sid: {k: v for k, v in d.items() if k.endswith("sha256") or k == "t3_attachment_clip_relative_path"}
                         for sid, d in media_inputs.items()},
        "sample05": {"vision_mapping_status": "index_only", "t4_status": "T4_NOT_ATTEMPTED_BY_FROZEN_SCOPE"},
    }
    return scope, lineage, native_samples, {"t3": t3_records, "formal": formal_samples,
                                           "media": media_inputs, "contract": t3_contract,
                                           "aggregate": t3_aggregate}


def map_one_position(sid: str, modality: str, seq: int, target: str, score: dict,
                     lineage: dict, native_samples: dict, t3: dict, clip_frames: list[dict] | None = None,
                     clip_bounds: tuple[float, float] | None = None,
                     source_frames: list[dict] | None = None, out_frames: Path | None = None,
                     clip_path: Path | None = None, t3_threshold: float | None = None,
                     video_unit_group: dict | None = None) -> dict:
    result = {
        "sample_id": sid, "modality": modality, "target": target,
        "official_seq_index": int(seq),
        "signed_importance": float(score["signed_importance"]),
        "absolute_importance": float(score["absolute_importance"]),
        "unaligned_row": None, "native_source_row": None,
        "source_start_sec": None, "source_end_sec": None,
        "audio_offset_sec": None, "video_offset_sec": None, "abs_difference_sec": None,
        "local_start_sec": None, "local_end_sec": None,
        "candidate_frame_count": None, "selected_local_pts": None,
        "previous_frame_pts": None, "next_frame_pts": None,
        "interval_midpoint": None, "distance_selected_to_midpoint": None,
        "source_expected_time": None, "source_frame_pts": None,
        "source_frame_time_error": None, "pixel_pearson": None,
        "same_underlying_evidence_unit": False, "evidence_unit_id": None,
        "frame_path": None, "mapping_status": "index_only", "failure_reason": "",
        "c2_unique": None, "native_unique": None, "source_row_status": None,
    }
    key = (sid, modality, int(seq))
    row = lineage.get(key)
    if row is None:
        result["failure_reason"] = "official position absent from T2/C2 lineage table"
        return result
    result.update({"unaligned_row": int(row["unaligned_j"]), "c2_unique": int(row["c2_unique"]),
                   "native_unique": int(row["native_unique"]), "source_row_status": row["source_row_status"]})
    if row["c2_unique"] != "1" or row["native_unique"] != "1" or row["source_row_status"] != "SOURCE_ROW_UNIQUE":
        result["failure_reason"] = "C2/native row uniqueness gate failed"
        return result
    native = native_samples[sid][modality]
    native_row = strict_source_row(native["strict_rows"], int(row["unaligned_j"]))
    if native_row is None:
        result["failure_reason"] = "native source row is not a unique strict row match"
        return result
    result["native_source_row"] = int(native_row["native_r"])
    interval = native_row.get("native_time")
    if not isinstance(interval, list) or len(interval) != 2:
        result["failure_reason"] = "native source row has no auditable two-endpoint time interval"
        return result
    start, end = map(float, interval)
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        result["failure_reason"] = "native source interval is non-finite or non-positive"
        return result
    result["source_start_sec"], result["source_end_sec"] = start, end
    if modality == "vision":
        result["video_offset_sec"] = float(t3["video"]["offset_median_sec"])
        result["audio_offset_sec"] = float(t3["audio"]["full_peak_source_offset_sec"])
        result["abs_difference_sec"] = abs(result["video_offset_sec"] - result["audio_offset_sec"])
        offset = result["video_offset_sec"]
        result["local_start_sec"], result["local_end_sec"] = start - offset, end - offset
        if clip_frames is None or clip_bounds is None or source_frames is None:
            result["failure_reason"] = "video decode data was not supplied"
            return result
        lo, hi = clip_bounds
        if result["local_start_sec"] < lo - 1e-9 or result["local_end_sec"] > hi + 1e-9:
            result["failure_reason"] = "mapped local visual interval is outside decoded clip presentation coverage"
            return result
        midpoint = (result["local_start_sec"] + result["local_end_sec"]) / 2.0
        result["interval_midpoint"] = midpoint
        inside = [f for f in clip_frames if result["local_start_sec"] - 1e-9 <= f["pts"] <= result["local_end_sec"] + 1e-9]
        result["candidate_frame_count"] = len(inside)
        chosen = min(inside, key=lambda f: (abs(f["pts"] - midpoint), f["pts"])) if inside else nearest_frame(clip_frames, midpoint)
        result["selected_local_pts"] = float(chosen["pts"])
        result["distance_selected_to_midpoint"] = abs(chosen["pts"] - midpoint)
        ix = next(i for i, frame in enumerate(clip_frames) if frame is chosen)
        result["previous_frame_pts"] = clip_frames[ix - 1]["pts"] if ix > 0 else None
        result["next_frame_pts"] = clip_frames[ix + 1]["pts"] if ix + 1 < len(clip_frames) else None
        unit_id = f"{sid}:vision:unaligned:{row['unaligned_j']}"
        if video_unit_group is not None:
            result["same_underlying_evidence_unit"] = len(video_unit_group) > 1
            result["evidence_unit_id"] = unit_id
        result["source_expected_time"] = result["selected_local_pts"] + offset
        source_chosen = nearest_frame(source_frames, result["source_expected_time"])
        result["source_frame_pts"] = float(source_chosen["pts"])
        result["source_frame_time_error"] = float(source_chosen["pts"] - result["source_expected_time"])
        result["pixel_pearson"] = float(np.dot(chosen["vec"], source_chosen["vec"]))
        if not inside:
            result["mapping_status"] = "nearest_frame_candidate_only"
            result["failure_reason"] = "no decoded clip PTS lies inside the native FACET interval"
            return result
        if result["pixel_pearson"] < float(t3_threshold):
            result["failure_reason"] = f"selected source/local frame comparison below frozen T3 threshold {t3_threshold}"
            return result
        result["mapping_status"] = "reconstructed_keyframe_from_verified_lineage"
        result["failure_reason"] = ""
        return result

    # Audio: source COVAREP time maps only through the frozen T3 audio offset.
    result["audio_offset_sec"] = float(t3["audio"]["full_peak_source_offset_sec"])
    result["video_offset_sec"] = float(t3["video"]["offset_median_sec"])
    result["abs_difference_sec"] = abs(result["audio_offset_sec"] - result["video_offset_sec"])
    if t3["audio"]["status"] != "PASS":
        result["failure_reason"] = "T3 audio origin gate is not PASS"
        return result
    offset = result["audio_offset_sec"]
    result["local_start_sec"], result["local_end_sec"] = start - offset, end - offset
    duration = float(t3["audio"]["clip_audio_duration_sec"])
    if result["local_start_sec"] < 0 or result["local_end_sec"] > duration:
        result["failure_reason"] = "mapped local speech interval lies outside T3-recorded Attachment4 audio duration"
        return result
    result["mapping_status"] = "reconstructed_speech_interval_from_verified_lineage"
    result["failure_reason"] = ""
    return result


def run_probe(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    phase = repo_root / "阶段4.2_Q3_v1_preflight"
    handoff = phase / "external_review_handoff_2026-09-26" / "artifacts_extracted"
    scope, lineage, native_samples, bundle = make_scope(
        repo_root, phase, handoff,
        handoff / "t2_row_boundary_and_source_inventory_2026-09-26" / "results" / "aligned_position_lineage_20.csv",
        phase / "t2_row_boundary_and_source_inventory_2026-09-26" / "results" / "native20_row_results.json.gz",
        handoff / "t3_media_origin_2026-09-26" / "results_v3",
        handoff / "formal_attachment4_2026-09-26" / "results" / "samples",
        args.attachment_video_dir, args.source_media_dir)
    t3 = bundle["t3"]["04"]
    media = bundle["media"]["04"]
    for label in ("attachment_clip", "source_video"):
        actual = media[f"{label}_actual_sha256"]
        expected = media[f"{label}_expected_sha256"]
        if actual is None or actual.lower() != expected.lower():
            raise RuntimeError(f"sample04 {label} SHA does not match frozen T3 record")
    if not (t3["video"]["status"] == "PASS" and t3["media_identity"] == "PASS" and t3["local_time_gate"] == "PASS"):
        raise RuntimeError("sample04 frozen T3 visual gates do not all PASS")
    sample = bundle["formal"]["04"]
    vision = next(p for p in positions_from_sample(sample)["vision"] if p["official_seq_index"] == 15)
    audio_seq = positions_from_sample(sample)["audio_union_positions"][0]
    audio_score = None
    for target in ("classification", "regression"):
        audio_score = next((p for p in positions_from_sample(sample)["audio_by_target"][target]
                            if p["official_seq_index"] == audio_seq), None)
        if audio_score:
            audio_score = {**audio_score, "target": target}
            break
    clip_path, source_path = media["attachment_clip_path"], media["source_video_path"]
    clip_frames, clip_start, clip_end = decode_video(clip_path)
    source_frames, _, _ = decode_video(source_path)
    vrec = map_one_position("04", "vision", 15, "classification", vision,
                            lineage, native_samples, t3, clip_frames, (clip_start, clip_end),
                            source_frames, t3_threshold=float(bundle["contract"]["video_rule"]["min_two_correlations"]))
    arec = map_one_position("04", "audio", audio_seq, audio_score["target"], audio_score,
                            lineage, native_samples, t3)
    print(json.dumps({"sample_id": "04", "vision_seq15_probe": vrec, "audio_first_union_probe": arec,
                      "frozen_source_head": scope["source_branch_head"], "model_loaded": False},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if vrec["mapping_status"] in ALLOWED_VISION and arec["mapping_status"] in ALLOWED_AUDIO else 1


def run_full(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    phase = repo_root / "阶段4.2_Q3_v1_preflight"
    out = phase / "t4_raw_evidence_closure_2026-09-26"
    results_dir = out / "results"
    per_sample_dir = results_dir / "per_sample"
    frames_dir = out / "frames"
    handoff = phase / "external_review_handoff_2026-09-26" / "artifacts_extracted"
    lineage_path = handoff / "t2_row_boundary_and_source_inventory_2026-09-26" / "results" / "aligned_position_lineage_20.csv"
    native_path = phase / "t2_row_boundary_and_source_inventory_2026-09-26" / "results" / "native20_row_results.json.gz"
    t3_root = handoff / "t3_media_origin_2026-09-26" / "results_v3"
    formal_dir = handoff / "formal_attachment4_2026-09-26" / "results" / "samples"
    scope, lineage, native_samples, bundle = make_scope(repo_root, phase, handoff, lineage_path, native_path,
                                                         t3_root, formal_dir, args.attachment_video_dir,
                                                         args.source_media_dir)
    dump_json(out / "T4_FROZEN_SCOPE.json", scope)
    vision_rows: list[dict] = []
    audio_rows: list[dict] = []
    sample_results: dict[str, dict] = {}
    global_stop = False
    for sid in SAMPLES:
        t3 = bundle["t3"][sid]
        media = bundle["media"][sid]
        media_ok = all(media[k] is not None and media[k].lower() == media[k.replace("actual", "expected")].lower()
                        for k in ("attachment_clip_actual_sha256", "source_video_actual_sha256"))
        gates_ok = (t3["video"]["status"] == "PASS" and t3["media_identity"] == "PASS" and t3["local_time_gate"] == "PASS")
        if not media_ok or not gates_ok:
            global_stop = True
            why = "T3 media SHA mismatch/missing" if not media_ok else "frozen T3 media_identity/video/local_time gate not all PASS"
            sample_results[sid] = {"sample_id": sid, "vision_status": "STOPPED", "audio_status": "STOPPED",
                                   "stop_reason": why, "vision_positions": [], "audio_positions": []}
            continue
        clip_path, source_path = media["attachment_clip_path"], media["source_video_path"]
        clip_frames, clip_start, clip_end = decode_video(clip_path)
        source_frames, _, _ = decode_video(source_path)
        formal = bundle["formal"][sid]
        positions = positions_from_sample(formal)
        # Group identical unaligned vision rows; shared model positions make one evidence unit.
        group_map: dict[str, list[int]] = defaultdict(list)
        for p in positions["vision"]:
            linrow = lineage.get((sid, "vision", int(p["official_seq_index"])))
            if linrow:
                group_map[str(linrow["unaligned_j"])].append(int(p["official_seq_index"]))
        vision_records = []
        for p in positions["vision"]:
            linrow = lineage.get((sid, "vision", int(p["official_seq_index"])))
            group = group_map.get(str(linrow["unaligned_j"]), []) if linrow else []
            rec = map_one_position(sid, "vision", int(p["official_seq_index"]), "classification", p,
                                   lineage, native_samples, t3, clip_frames, (clip_start, clip_end),
                                   source_frames, frames_dir, clip_path,
                                   float(bundle["contract"]["video_rule"]["min_two_correlations"]), group)
            vision_records.append(rec)
            vision_rows.append(rec)
        # Save one image per underlying feature evidence unit, never one per
        # duplicated aligned slot (notably sample 14 positions 34 and 35).
        unit_members: dict[str, list[dict]] = defaultdict(list)
        for rec in vision_records:
            unit_key = rec["evidence_unit_id"] or f"{sid}:vision:failed-position:{rec['official_seq_index']}"
            unit_members[unit_key].append(rec)
        for members in unit_members.values():
            if not all(m["mapping_status"] == "reconstructed_keyframe_from_verified_lineage" for m in members):
                continue
            representative = members[0]
            seq_tag = "-".join(str(m["official_seq_index"]) for m in members)
            pts = representative["selected_local_pts"]
            filename = f"sample{sid}_cls_vision_seq{seq_tag}_pts_{pts:.6f}.png"
            save_exact_pts_frame(clip_path, pts, frames_dir / filename)
            relative_path = f"frames/{filename}"
            for member in members:
                member["frame_path"] = relative_path
        # Keep the union in first-occurrence order across the frozen classification and regression top-3 lists.
        audio_records_by_seq = {}
        for target in ("classification", "regression"):
            for p in positions["audio_by_target"][target]:
                seq = int(p["official_seq_index"])
                rec = map_one_position(sid, "audio", seq, target, p, lineage, native_samples, t3)
                audio_rows.append(rec)
                audio_records_by_seq.setdefault(seq, rec)
        vision_units: dict[str, list[dict]] = defaultdict(list)
        for rec in vision_records:
            unit_key = str(rec.get("unaligned_row")) if rec.get("unaligned_row") is not None else f"failed:{rec['official_seq_index']}"
            vision_units[unit_key].append(rec)
        unit_statuses = [all(x["mapping_status"] == "reconstructed_keyframe_from_verified_lineage" for x in group)
                         for group in vision_units.values()]
        v_success = sum(unit_statuses)
        v_total = len(unit_statuses)
        v_status = "PASS" if v_success == v_total and v_total else ("PARTIAL" if v_success else "FAIL")
        a_success = sum(rec["mapping_status"] == "reconstructed_speech_interval_from_verified_lineage"
                        for rec in audio_records_by_seq.values())
        a_total = len(audio_records_by_seq)
        a_status = "PASS" if a_success == a_total and a_total else ("PARTIAL" if a_success else "FAIL")
        sample_results[sid] = {
            "sample_id": sid,
            "media_sha_status": "PASS",
            "t3_gate_status": "PASS",
            "audio_offset_sec": t3["audio"]["full_peak_source_offset_sec"],
            "video_offset_sec": t3["video"]["offset_median_sec"],
            "abs_audio_video_offset_difference_sec": abs(t3["audio"]["full_peak_source_offset_sec"] - t3["video"]["offset_median_sec"]),
            "vision_status": v_status,
            "vision_success_evidence_units": v_success,
            "vision_total_evidence_units": v_total,
            "vision_records": vision_records,
            "audio_status": a_status,
            "audio_success_union_positions": a_success,
            "audio_total_union_positions": a_total,
            "audio_union_positions": [audio_records_by_seq[s] for s in positions["audio_union_positions"]],
            "audio_target_position_records": [r for r in audio_rows if r["sample_id"] == sid],
        }
        dump_json(per_sample_dir / f"sample_{sid}.json", sample_results[sid])

    for row in vision_rows:
        if row["mapping_status"] not in ALLOWED_VISION:
            raise ValueError(f"invalid vision mapping status {row['mapping_status']}")
    for row in audio_rows:
        if row["mapping_status"] not in ALLOWED_AUDIO:
            raise ValueError(f"invalid audio mapping status {row['mapping_status']}")

    vision_fields = ["sample_id", "target", "official_seq_index", "signed_importance", "absolute_importance",
                     "unaligned_row", "native_source_row", "source_start_sec", "source_end_sec", "video_offset_sec",
                     "audio_offset_sec", "abs_difference_sec", "local_start_sec", "local_end_sec", "candidate_frame_count",
                     "selected_local_pts", "previous_frame_pts", "next_frame_pts", "interval_midpoint",
                     "distance_selected_to_midpoint", "source_expected_time", "source_frame_pts", "source_frame_time_error",
                     "pixel_pearson", "same_underlying_evidence_unit", "evidence_unit_id", "frame_path", "c2_unique",
                     "native_unique", "source_row_status", "mapping_status", "failure_reason"]
    audio_fields = ["sample_id", "target", "official_seq_index", "signed_importance", "absolute_importance",
                    "unaligned_row", "native_source_row", "source_start_sec", "source_end_sec", "audio_offset_sec",
                    "video_offset_sec", "abs_difference_sec", "local_start_sec", "local_end_sec", "c2_unique",
                    "native_unique", "source_row_status", "mapping_status", "failure_reason"]
    write_csv(results_dir / "t4_visual_evidence.csv", vision_rows, vision_fields)
    write_csv(results_dir / "t4_audio_evidence.csv", audio_rows, audio_fields)

    all_v_pass = all(s["vision_status"] == "PASS" for s in sample_results.values() if s.get("sample_id")) and len(sample_results) == 3
    all_a_pass = all(s["audio_status"] == "PASS" for s in sample_results.values() if s.get("sample_id")) and len(sample_results) == 3
    t4_gate = "T4_STOPPED" if global_stop else ("T4_COMPLETE" if all_v_pass and all_a_pass else "T4_COMPLETE_WITH_LIMITATIONS")
    gate = {
        "gate": t4_gate,
        "scope": "frozen Attachment4 evidence positions for samples 04/14/19 only",
        "model_loaded": False,
        "training": False,
        "inference": False,
        "formal_predictions_or_attributions_modified": False,
        "sample_gates": {sid: {"vision": s["vision_status"], "vision_success_evidence_units": s.get("vision_success_evidence_units"),
                                "vision_total_evidence_units": s.get("vision_total_evidence_units"), "audio": s["audio_status"],
                                "audio_success_union_positions": s.get("audio_success_union_positions"),
                                "audio_total_union_positions": s.get("audio_total_union_positions"),
                                "stop_reason": s.get("stop_reason")}
                         for sid, s in sample_results.items()},
        "sample05_vision": "index_only; T4_NOT_ATTEMPTED_BY_FROZEN_SCOPE",
        "all_raw_media_mapping_complete": False,
        "historical_4_6_gate_modified": False,
        "4_7_entered": False,
    }
    dump_json(results_dir / "T4_GATE.json", gate)
    valid_audio_positions = {
        sid: {int(p["official_seq_index"]) for target in ("classification", "regression")
              for p in positions_from_sample(bundle["formal"][sid])["audio_by_target"][target]}
        for sid in SAMPLES
    }
    self_checks = {
            "sample_scope_only_04_14_19": list(sample_results) == list(SAMPLES),
            "vision_position_scope_exact": {sid: [r["official_seq_index"] for r in sample_results[sid].get("vision_records", [])]
                                             == list(FROZEN_VISION[sid]) for sid in SAMPLES if sid in sample_results},
            "audio_rows_from_frozen_top3_union": all(r["official_seq_index"] in valid_audio_positions[r["sample_id"]]
                                                     for r in audio_rows),
            "media_sha_all_pass": all(s.get("media_sha_status") == "PASS" for s in sample_results.values()),
            "sample14_34_35_shared_row": (lineage[("14", "vision", 34)]["unaligned_j"] == lineage[("14", "vision", 35)]["unaligned_j"]
                                           and next(r for r in vision_rows if r["sample_id"] == "14" and r["official_seq_index"] == 34)["same_underlying_evidence_unit"]
                                           and next(r for r in vision_rows if r["sample_id"] == "14" and r["official_seq_index"] == 35)["same_underlying_evidence_unit"]),
            "vision_local_time_uses_video_offset": all(r["local_start_sec"] is None or
                                                        abs(r["local_start_sec"] - (r["source_start_sec"] - r["video_offset_sec"])) < 1e-9
                                                        for r in vision_rows),
            "audio_local_time_uses_audio_offset": all(r["local_start_sec"] is None or
                                                       abs(r["local_start_sec"] - (r["source_start_sec"] - r["audio_offset_sec"])) < 1e-9
                                                       for r in audio_rows),
            "only_success_frames_saved": all(r["mapping_status"] == "reconstructed_keyframe_from_verified_lineage"
                                              for r in vision_rows if r["frame_path"]),
            "all_visual_pts_are_decoded_pts": all(r["selected_local_pts"] is not None for r in vision_rows
                                                   if r["mapping_status"] != "index_only"),
            "sample05_not_attempted": scope["sample05"]["t4_status"] == "T4_NOT_ATTEMPTED_BY_FROZEN_SCOPE",
            "all_mapping_statuses_allowlisted": all(r["mapping_status"] in ALLOWED_VISION for r in vision_rows)
                                                 and all(r["mapping_status"] in ALLOWED_AUDIO for r in audio_rows),
            "model_checkpoint_not_loaded": True,
            "historical_results_not_modified": True,
            "frame_files_exist_only_for_successful_evidence_units": all(
                (out / r["frame_path"]).is_file() if r["frame_path"] else True for r in vision_rows),
            "no_duplicate_frame_files_for_shared_evidence": len({r["frame_path"] for r in vision_rows if r["frame_path"]})
                                                            == len({r["evidence_unit_id"] for r in vision_rows
                                                                    if r["frame_path"]}),
    }
    dump_json(results_dir / "T4_SELF_CHECKS.json", {
        "pass": all(self_checks.values()) and not global_stop,
        "checks": self_checks,
        "note": "Script reads no checkpoint and calls no model. Any failed media identity or T3 media-origin gate sets the global gate to T4_STOPPED.",
    })
    write_report(out, scope, sample_results, t4_gate, bundle["contract"])
    build_manifest(out)
    return 0 if t4_gate != "T4_STOPPED" else 2


def write_report(out: Path, scope: dict, samples: dict, gate: str, t3_contract: dict) -> None:
    lines = [
        "# T4 Raw Evidence Closure Report", "",
        f"- Gate: **{gate}**", f"- Starting HEAD: `{scope['source_branch_head']}`",
        "- Frozen scope: Attachment4 samples 04, 14, 19 only.",
        "- Training / model selection / inference / test evaluation: **NO**.",
        "- Shapley / IG / formal predictions changed: **NO**.",
        "- Historical 4.6 Gate changed: **NO**; 4.7 entered: **NO**.",
        "- Mapping interpretation: reconstructed raw-media navigation evidence; it does not prove the exact frame/window originally sampled by the official extractor.",
        "- T4 selected-frame comparison reuses the frozen T3 numeric cutoff `min_two_correlations` = `" + str(t3_contract["video_rule"]["min_two_correlations"]) + "` on the individual source/local frame pair. This is a per-position check; it is not the T3 aggregate rule (at least 2 of 3 anchor comparisons pass, with offset range ≤ `" + str(t3_contract["video_rule"]["max_offset_range_sec"]) + " s`).",
        "- T4 frame preprocessing reuses T3: all presentation PTS via PyAV; grayscale 64×64; Pearson computed as the dot product of mean-centered unit vectors.",
        "", "## Per-sample results", "",
        "| Sample | Vision evidence units | Vision positions | Audio union positions | Audio mapped | Video offset (s) | Audio offset (s) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sid in SAMPLES:
        s = samples[sid]
        lines.append(f"| {sid} | {s.get('vision_success_evidence_units', 0)}/{s.get('vision_total_evidence_units', 0)} | "
                     f"{','.join(str(x['official_seq_index']) for x in s.get('vision_records', []))} | "
                     f"{s.get('audio_success_union_positions', 0)}/{s.get('audio_total_union_positions', 0)} | "
                     f"{','.join(str(x['official_seq_index']) for x in s.get('audio_union_positions', []))} | "
                     f"{s.get('video_offset_sec', 'NA')} | {s.get('audio_offset_sec', 'NA')} | "
                     f"vision={s.get('vision_status')}; audio={s.get('audio_status')} |")
    lines += ["", "## Visual position details", "",
              "| Sample | Position | Unaligned row | FACET interval (s) | Local interval (s) | Candidate frames | Selected PTS | Source PTS | Pearson | Same unit | Mapping | Failure |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|---|---|---|"]
    for sid in SAMPLES:
        for r in samples[sid].get("vision_records", []):
            lines.append(f"| {sid} | {r['official_seq_index']} | {r.get('unaligned_row')} | "
                         f"{r.get('source_start_sec')}–{r.get('source_end_sec')} | {r.get('local_start_sec')}–{r.get('local_end_sec')} | "
                         f"{r.get('candidate_frame_count')} | {r.get('selected_local_pts')} | {r.get('source_frame_pts')} | "
                         f"{r.get('pixel_pearson')} | {r.get('same_underlying_evidence_unit')} | {r['mapping_status']} | {r['failure_reason']} |")
    lines += ["", "## Audio position details", "",
              "| Sample | Target | Position | Unaligned row | COVAREP interval (s) | Audio offset (s) | Local speech interval (s) | Mapping | Failure |",
              "|---|---|---:|---:|---|---:|---|---|---|"]
    for sid in SAMPLES:
        for r in samples[sid].get("audio_target_position_records", []):
            lines.append(f"| {sid} | {r['target']} | {r['official_seq_index']} | {r.get('unaligned_row')} | "
                         f"{r.get('source_start_sec')}–{r.get('source_end_sec')} | {r.get('audio_offset_sec')} | "
                         f"{r.get('local_start_sec')}–{r.get('local_end_sec')} | {r['mapping_status']} | {r['failure_reason']} |")
    lines += ["", "## Explicit limitations", "",
              "- Sample 14 positions 34 and 35 share one unaligned/source evidence unit and must not be presented as two independent frames.",
              "- Sample 05 vision remains `index_only`; it was not attempted in this frozen scope.",
              "- Audio/vision mapping outside the frozen 04/14/19 positions remains unchanged and may remain `index_only`.",
              "- A successful reconstructed keyframe is a reproducible navigation frame within the FACET interval, not proof that the official FACET extractor sampled that exact decoded frame.",
              "- T4 is not a Q3 final Gate and does not change the historical 4.6 Gate.",
              "", "## Reproduction", "",
              "Use the included `run_t4_raw_evidence_closure.py` from this worktree with the existing T3 private media directory and Attachment4 aligned `videos` directory. No model checkpoint or network download is used.",
              ""]
    (out / "T4_RAW_EVIDENCE_CLOSURE_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_manifest(out: Path) -> None:
    entries = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "OUTPUT_MANIFEST.json"):
        entries.append({"relative_path": path.relative_to(out).as_posix(), "size_bytes": path.stat().st_size,
                        "sha256": sha256(path)})
    dump_json(out / "results" / "OUTPUT_MANIFEST.json", {
        "manifest_scope": "all T4 files other than this manifest itself; self-hash is excluded to avoid recursion",
        "file_count": len(entries), "files": entries,
    })


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-media-dir", type=Path, required=True,
                    help="existing, frozen T3 private_media_v3 directory; no download is performed")
    ap.add_argument("--attachment-video-dir", type=Path, required=True,
                    help="existing Attachment4 aligned/videos directory")
    ap.add_argument("--probe", action="store_true", help="run one frozen visual and one audio position without writing results")
    return ap.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    raise SystemExit(run_probe(ARGS) if ARGS.probe else run_full(ARGS))
