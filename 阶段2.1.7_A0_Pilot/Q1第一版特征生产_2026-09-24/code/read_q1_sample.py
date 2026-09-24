from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_reader import load_q1_feature, summarize_q1_feature


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and summarize one q1-feature-v1.0 NPZ without pickle.")
    parser.add_argument("npz", type=Path, help="Path to one per-sample NPZ")
    args = parser.parse_args()
    arrays = load_q1_feature(args.npz)
    print(json.dumps(summarize_q1_feature(arrays), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
