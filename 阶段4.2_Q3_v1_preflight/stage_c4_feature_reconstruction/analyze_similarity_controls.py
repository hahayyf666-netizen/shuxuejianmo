"""Check whether media proxy similarities depend on temporal row order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from reconstruct_media import linear_cka


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unaligned", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for number in range(1, 21):
        sid = f"{number:02d}"
        with (args.unaligned / f"{sid}.pkl").open("rb") as stream:
            item = pickle.load(stream)
        with np.load(args.candidate_dir / f"{sid}.npz", allow_pickle=False) as cache:
            audio_len = int(item["audio_lengths"])
            vision_len = int(item["vision_lengths"])
            official_audio = np.asarray(item["audio"][:audio_len])
            candidate_audio = np.asarray(cache["audio_candidate_25D"])
            if len(candidate_audio) != audio_len:
                raise ValueError(f"{sid}: audio candidate length mismatch")
            audio = {
                "observed_CKA": linear_cka(official_audio, candidate_audio),
                "reversed_CKA": linear_cka(official_audio, candidate_audio[::-1]),
            }
            official_vision = np.asarray(item["vision"][:vision_len])
            candidate_vision = np.asarray(cache["vision_candidate_52D"])
            valid = np.asarray(cache["vision_candidate_face_valid"], dtype=bool)
            pts = np.asarray(cache["vision_candidate_pts_sec"])
            mask = valid & np.isfinite(pts)
            visual_target = official_vision[mask]
            visual_candidate = candidate_vision[mask]
            vision = {
                "observed_CKA": linear_cka(visual_target, visual_candidate),
                "reversed_CKA": linear_cka(visual_target, visual_candidate[::-1]),
                "face_valid_rows": int(np.count_nonzero(mask)),
            }
            records.append({"sample_id": sid, "audio": audio, "vision": vision})
    summary = {}
    for modality in ("audio", "vision"):
        pairs = [(row[modality]["observed_CKA"], row[modality]["reversed_CKA"])
                 for row in records if row[modality]["observed_CKA"] is not None
                 and row[modality]["reversed_CKA"] is not None]
        summary[modality] = {
            "comparable_samples": len(pairs),
            "median_observed_CKA": float(np.median([x for x, _ in pairs])) if pairs else None,
            "median_reversed_CKA": float(np.median([y for _, y in pairs])) if pairs else None,
            "observed_above_reversed_samples": sum(x > y for x, y in pairs),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_sample": records},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
