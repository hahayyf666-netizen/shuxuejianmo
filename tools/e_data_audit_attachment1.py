from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(".")
OUT = ROOT / "_e_data_audit_out"
OUT.mkdir(exist_ok=True)
SPEC_MIN_DURATION = 2.648


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_attachment1() -> Path:
    candidates = sorted(p for p in (ROOT / "E题数据").glob("附件1*") if p.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one attachment1 directory, got: {candidates}")
    return candidates[0]


def norm_clip_id(x) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path)
    ]
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"ok": True, "data": json.loads(p.stdout)}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def main() -> None:
    root = find_attachment1()
    xlsx = sorted(root.rglob("label-100.xlsx"))
    if len(xlsx) != 1:
        raise RuntimeError(f"Expected one label-100.xlsx, got {xlsx}")
    df = pd.read_excel(xlsx[0])
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    missing_cols = sorted(required - set(df.columns))

    label_rows = []
    for _, r in df.iterrows():
        vid = str(r["video_id"]).strip()
        cid = norm_clip_id(r["clip_id"])
        label_rows.append({
            "video_id": vid,
            "clip_id": cid,
            "key": vid + "$_$" + cid,
            "text": str(r["text"]),
            "label": float(r["label"]),
            "annotation": str(r["annotation"]),
        })
    label_keys = [x["key"] for x in label_rows]

    videos = sorted(root.rglob("*.mp4"))
    media = []
    digest_groups = defaultdict(list)
    for p in videos:
        vid = p.parent.name
        cid = p.stem
        key = vid + "$_$" + cid
        digest = sha256_file(p)
        digest_groups[digest].append(key)
        pr = ffprobe(p)
        item = {"key": key, "path": p.as_posix(), "sha256": digest, "ffprobe_ok": pr["ok"]}
        if pr["ok"]:
            d = pr["data"]
            streams = d.get("streams", [])
            vstreams = [s for s in streams if s.get("codec_type") == "video"]
            astreams = [s for s in streams if s.get("codec_type") == "audio"]
            fmt = d.get("format", {})
            def safe_float(v):
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            format_duration = safe_float(fmt.get("duration"))
            video_stream_duration = safe_float(vstreams[0].get("duration")) if vstreams else None
            audio_stream_duration = safe_float(astreams[0].get("duration")) if astreams else None
            item.update({
                "ffprobe_format_duration_sec": format_duration,
                "video_stream_duration_sec": video_stream_duration,
                "audio_stream_duration_sec": audio_stream_duration,
                "has_audio": bool(astreams),
                "video_codec": vstreams[0].get("codec_name") if vstreams else None,
                "width": vstreams[0].get("width") if vstreams else None,
                "height": vstreams[0].get("height") if vstreams else None,
                "r_frame_rate": vstreams[0].get("r_frame_rate") if vstreams else None,
                "audio_codec": astreams[0].get("codec_name") if astreams else None,
                "audio_sample_rate": astreams[0].get("sample_rate") if astreams else None,
                "audio_channels": astreams[0].get("channels") if astreams else None,
            })
        else:
            item["error"] = pr["error"]
        media.append(item)

    video_keys = [x["key"] for x in media]
    label_set, video_set = set(label_keys), set(video_keys)
    duplicate_label_keys = [k for k, n in Counter(label_keys).items() if n > 1]
    duplicate_video_keys = [k for k, n in Counter(video_keys).items() if n > 1]
    duplicate_text = [t for t, n in Counter(x["text"] for x in label_rows).items() if n > 1]
    duplicate_video_bytes = [v for v in digest_groups.values() if len(v) > 1]
    short_presentations = [
        {
            "key": x["key"],
            "ffprobe_format_duration_sec": x.get("ffprobe_format_duration_sec"),
            "video_stream_duration_sec": x.get("video_stream_duration_sec"),
            "audio_stream_duration_sec": x.get("audio_stream_duration_sec"),
        }
        for x in media
        if x.get("ffprobe_format_duration_sec") is not None
        and x["ffprobe_format_duration_sec"] < SPEC_MIN_DURATION
    ]

    report = {
        "attachment": "attachment1",
        "root": root.as_posix(),
        "label_file": xlsx[0].as_posix(),
        "counts": {
            "videos": len(videos),
            "label_rows": len(df),
            "unique_video_ids": len(set(x["video_id"] for x in label_rows)),
        },
        "columns": list(map(str, df.columns)),
        "missing_required_columns": missing_cols,
        "label_nulls": {str(k): int(v) for k, v in df.isna().sum().items()},
        "label_distribution": dict(Counter(x["annotation"] for x in label_rows)),
        "continuous_label_range": [
            min(x["label"] for x in label_rows),
            max(x["label"] for x in label_rows),
        ],
        "duplicate_label_keys": duplicate_label_keys,
        "duplicate_video_keys": duplicate_video_keys,
        "duplicate_text_count": len(duplicate_text),
        "duplicate_video_byte_groups": duplicate_video_bytes,
        "labels_without_video": sorted(label_set - video_set),
        "videos_without_label": sorted(video_set - label_set),
        "ffprobe_failures": [x for x in media if not x["ffprobe_ok"]],
        "videos_without_audio": [x["key"] for x in media if x.get("ffprobe_ok") and not x.get("has_audio")],
        "ffprobe_format_duration_below_problem_stated_minimum": short_presentations,
        "media_summary": {
            "video_codecs": dict(Counter(x.get("video_codec") for x in media if x.get("ffprobe_ok"))),
            "audio_codecs": dict(Counter(x.get("audio_codec") for x in media if x.get("ffprobe_ok"))),
            "resolutions": dict(Counter(
                f"{x.get('width')}x{x.get('height')}" for x in media if x.get("ffprobe_ok")
            )),
            "frame_rates": dict(Counter(x.get("r_frame_rate") for x in media if x.get("ffprobe_ok"))),
            "audio_sample_rates": dict(Counter(x.get("audio_sample_rate") for x in media if x.get("ffprobe_ok"))),
            "audio_channels": dict(Counter(str(x.get("audio_channels")) for x in media if x.get("ffprobe_ok"))),
        },
    }
    hard_failures = []
    if len(videos) != 100:
        hard_failures.append(f"video_count={len(videos)} expected=100")
    if len(df) != 100:
        hard_failures.append(f"label_rows={len(df)} expected=100")
    if missing_cols:
        hard_failures.append(f"missing_columns={missing_cols}")
    if report["labels_without_video"] or report["videos_without_label"]:
        hard_failures.append("label/video key mapping is not one-to-one")
    if duplicate_label_keys or duplicate_video_keys:
        hard_failures.append("duplicate sample keys exist")
    if report["ffprobe_failures"]:
        hard_failures.append("one or more videos are unreadable")
    if report["videos_without_audio"]:
        hard_failures.append("one or more videos have no audio stream")
    report["hard_failures"] = hard_failures
    report["status"] = "PASS" if not hard_failures else "FAIL"

    (OUT / "attachment1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "E题附件1数据审计",
        "=" * 60,
        f"status={report['status']}",
        f"videos={len(videos)} label_rows={len(df)} unique_video_ids={report['counts']['unique_video_ids']}",
        f"labels_without_video={len(report['labels_without_video'])} videos_without_label={len(report['videos_without_label'])}",
        f"duplicate_label_keys={len(duplicate_label_keys)} duplicate_video_keys={len(duplicate_video_keys)}",
        f"duplicate_text_count={len(duplicate_text)} duplicate_video_byte_groups={len(duplicate_video_bytes)}",
        f"ffprobe_failures={len(report['ffprobe_failures'])} videos_without_audio={len(report['videos_without_audio'])}",
        f"label_distribution={report['label_distribution']}",
        f"continuous_label_range={report['continuous_label_range']}",
        f"ffprobe_format_duration_below_2.648s={short_presentations}",
        f"media_summary={report['media_summary']}",
        f"hard_failures={hard_failures}",
    ]
    (OUT / "attachment1_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
