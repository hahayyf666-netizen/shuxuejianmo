from __future__ import annotations

import csv
import json
from pathlib import Path

CANDIDATE = Path(__file__).resolve().parents[1]
STAGE = CANDIDATE.parent
REPO = STAGE.parent
RESULTS = CANDIDATE / "results"
CLASS_NAMES = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_indices(row: dict[str, str], target: str, modality: str) -> list[int]:
    prefix = "cls" if target == "classification" else "reg"
    value = row.get(f"top_seq_{prefix}_{modality}", "")
    return [int(part) for part in value.split(";") if part.strip()]


def phi_values(row: dict[str, str], target: str) -> dict[str, float]:
    prefix = "phi_cls" if target == "classification" else "phi_reg"
    return {m: float(row[f"{prefix}_{m}"]) for m in MODALITIES}


def max_positive(values: dict[str, float]) -> str | None:
    positive = {k: v for k, v in values.items() if v > 0.0}
    if not positive:
        return None
    return max(MODALITIES, key=lambda m: (positive.get(m, float("-inf")), -MODALITIES.index(m)))


def max_absolute(values: dict[str, float]) -> str:
    return max(MODALITIES, key=lambda m: (abs(values[m]), -MODALITIES.index(m)))
