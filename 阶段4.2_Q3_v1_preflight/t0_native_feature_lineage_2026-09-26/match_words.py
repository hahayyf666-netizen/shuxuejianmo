"""Locate frozen Attachment 4 transcript candidates in a timestamped word CSD.

Transcript matching only identifies a candidate source video and time offset.
It does not validate or assign audio/vision feature row timestamps.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pickle
import re
from pathlib import Path

import h5py


HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "results" / "t0_frozen_contract.json").read_text(encoding="utf-8"))
WORDS = HERE.parents[2] / "q3_native_t0_assets" / "CMU_MOSEI_TimestampedWords.csd"


def normalize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def main() -> None:
    targets = {}
    for item in CONTRACT["sources"]:
        with Path(item["files"]["unaligned_pkl"]["path"]).open("rb") as source:
            record = pickle.load(source)
        raw_text = str(record["raw_text"])
        targets[item["sample_id"]] = {"raw_text": raw_text, "tokens": normalize(raw_text)}
    all_words: dict[str, list[tuple[str, float, float]]] = {}
    gram_index: dict[tuple[str, ...], list[tuple[str, int]]] = collections.defaultdict(list)
    with h5py.File(WORDS, "r") as csd:
        data = csd["words/data"]
        for video_id in data:
            features = data[video_id]["features"][:].reshape(-1)
            intervals = data[video_id]["intervals"][:]
            normalized = []
            for value, (start, end) in zip(features, intervals):
                tokens = normalize(value.decode("utf-8", errors="replace"))
                for token in tokens:
                    if token != "sp":
                        normalized.append((token, float(start), float(end)))
            all_words[video_id] = normalized
            for position in range(max(0, len(normalized) - 2)):
                gram = tuple(row[0] for row in normalized[position:position + 3])
                gram_index[gram].append((video_id, position))
    output = []
    for sample_id, target in targets.items():
        tokens = target["tokens"]
        offsets: collections.Counter[tuple[str, int]] = collections.Counter()
        for text_pos in range(max(0, len(tokens) - 2)):
            for video_id, video_pos in gram_index.get(tuple(tokens[text_pos:text_pos + 3]), []):
                offsets[(video_id, video_pos - text_pos)] += 1
        best = []
        for (video_id, offset), gram_hits in offsets.most_common(10):
            source = all_words[video_id]
            exact_positions = [(t, offset + t) for t in range(len(tokens))
                               if 0 <= offset + t < len(source) and tokens[t] == source[offset + t][0]]
            if exact_positions:
                start = source[exact_positions[0][1]][1]
                end = source[exact_positions[-1][1]][2]
            else:
                start = end = None
            best.append({"source_video_id": video_id, "word_offset": offset,
                         "trigram_hits": gram_hits, "exact_ordered_token_hits": len(exact_positions),
                         "target_token_count": len(tokens), "source_start_sec": start,
                         "source_end_sec": end,
                         "source_phrase_excerpt": " ".join(row[0] for row in source[max(0, offset):max(0, offset) + len(tokens)])})
        output.append({"sample_id": sample_id, "raw_text": target["raw_text"],
                       "normalized_target_tokens": tokens, "top_candidates": best[:5],
                       "candidate_video_id_status": "candidate_only_not_feature_verified"})
    result = {"timestamped_words_sha256": hashlib.file_digest(WORDS.open("rb"), "sha256").hexdigest(),
              "source_group_count": len(all_words), "samples": output}
    path = HERE / "results" / "t0_transcript_source_candidates.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "candidates": [
        {"sample_id": row["sample_id"], "top": row["top_candidates"][:1]}
        for row in output]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
