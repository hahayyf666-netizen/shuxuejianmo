from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from common import CANDIDATE, RESULTS, STAGE, read_csv, read_json

HIST_RESULTS = STAGE / "external_review_handoff_2026-09-26/artifacts_extracted/formal_attachment4_2026-09-26/results"
T4_RESULTS = STAGE / "t4_raw_evidence_closure_2026-09-26/results"
T4_FRAMES = T4_RESULTS.parent / "frames"


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.2, "grid.linewidth": 0.6,
    })


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt_prob(row):
    return f"P(Negative)={float(row['p_negative']):.3f}   P(Neutral)={float(row['p_neutral']):.3f}   P(Positive)={float(row['p_positive']):.3f}"


def modality_bars(ax, vals):
    labels = ["Text", "Audio", "Vision"]
    colors = ["#3977A8" if v >= 0 else "#C05A59" for v in vals]
    ax.barh(labels, vals, color=colors, edgecolor="white", height=.56)
    ax.axvline(0, color="#3d4650", linewidth=.8)
    ax.set_xlabel("Shapley contribution to predicted-class probability")
    ax.invert_yaxis()
    low = min(0.0, min(vals)); high = max(0.0, max(vals))
    span = max(high - low, 0.05); pad = max(.12 * span, .015)
    ax.set_xlim(low - pad, high + pad)
    for y, value in enumerate(vals):
        ax.text(value + (0.008 if value >= 0 else -0.008), y, f"{value:+.3f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9, clip_on=False)


def make_case_card(sid, row, sample, out, frame_paths=None, anomaly=False):
    has_frames = bool(frame_paths)
    height_ratios = [0.55, 2.2, 1.35, 0.28] if has_frames else [0.55, 2.2, 0.82]
    fig = plt.figure(figsize=(10.6, 7.4 if has_frames else 6.8), constrained_layout=True)
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)
    primary = row["primary_reference_modality_cls"]
    dominant = row["dominant_influence_modality_cls"]
    if anomaly:
        title = "Sample 13 — Anomaly / Boundary Case"
    elif sid == "04":
        title = f"Sample 04 — {row['predicted_class']}; {primary}-primary reference"
    elif sid == "03":
        title = f"Sample 03 — {row['predicted_class']}; {primary}-primary reference"
    elif sid == "15":
        confidence = max(float(row[f"p_{name.lower()}"]) for name in ("Negative", "Neutral", "Positive"))
        title = f"Sample 15 — High-confidence {row['predicted_class']} (p={confidence:.3f}); {primary}-primary reference"
    else:
        title = f"Frozen B0 explanation card — Sample {sid}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_info = fig.add_subplot(gs[0]); ax_info.axis("off")
    ax_info.text(0.01, .68, f"Prediction: {row['predicted_class']}   |   Intensity: {float(row['predicted_intensity']):+.3f}",
                 fontsize=10.5, fontweight="bold")
    ax_info.text(0.01, .18, fmt_prob(row), fontsize=9.5)
    ax = fig.add_subplot(gs[1])
    vals = [float(row[f"phi_cls_{m}"]) for m in ("text", "audio", "vision")]
    modality_bars(ax, vals)
    if anomaly:
        ax.set_title("Signed classification Shapley (feature-space response)", loc="left", fontsize=9.5)
    else:
        ax.set_title(f"Signed Shapley (+ supports predicted class; − opposes it) | primary reference: {primary}; dominant influence: {dominant}",
                     loc="left", fontsize=9.2)
    if has_frames:
        frame_grid = gs[2].subgridspec(1, min(3, len(frame_paths)), wspace=.04)
        for i, (path, pts) in enumerate(frame_paths[:3]):
            image_ax = fig.add_subplot(frame_grid[0, i])
            image_ax.imshow(mpimg.imread(path))
            image_ax.set_title(f"Reconstructed keyframe\nlocal PTS {pts:.3f} s", fontsize=8.5)
            image_ax.axis("off")
        note_ax = fig.add_subplot(gs[3]); note_ax.axis("off")
        note_ax.text(.01, .35, "T4-verified lineage; not claimed as the official extractor's exact sampled frame",
                     fontsize=8.2, color="#3e4c59")
    elif anomaly:
        note_ax = fig.add_subplot(gs[2]); note_ax.axis("off")
        note_ax.text(.02, .78, "aligned visual features are all-zero", fontsize=9.5, color="#8b2525")
        note_ax.text(.02, .48, "raw visual evidence unavailable", fontsize=9.5, color="#8b2525")
        note_ax.text(.02, .18, "vision attribution is feature-branch attribution only", fontsize=9.5, color="#8b2525")
    else:
        text_ax = fig.add_subplot(gs[2]); text_ax.axis("off")
        text_ax.text(.01, .88, "Verified text spans (top |conditional IG|; signed values retained)", fontsize=8.8, fontweight="bold")
        local = sample["targets"]["classification"]["modalities"]["text"]["top_positions"][:4]
        for k, item in enumerate(local):
            col, line = k % 2, k // 2
            x = .01 + col * .50; y = .57 - line * .38
            snippet = str(item.get("raw_text_substring", ""))
            span = f"{snippet!r} ({item.get('char_start')}:{item.get('char_end')}, IG {float(item['signed_importance']):+.3f})"
            text_ax.text(x, y, span, fontsize=8.2, ha="left", va="center")
    save(fig, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=CANDIDATE / "figures")
    ap.add_argument("--historical-results", type=Path, default=HIST_RESULTS)
    ap.add_argument("--t4-results", type=Path, default=T4_RESULTS)
    args = ap.parse_args()
    set_style()
    final_rows = {r["sample_id"]: r for r in read_csv(args.results / "attachment4_predictions_explanations_final.csv")}
    hist_dir = args.historical_results / "samples"
    samples = {sid: read_json(hist_dir / f"sample_{sid}.json") for sid in (f"{i:02d}" for i in range(1, 21))}
    visual = read_csv(args.t4_results / "t4_visual_evidence.csv")
    visual04 = [r for r in visual if r["sample_id"] == "04" and r["mapping_status"] == "reconstructed_keyframe_from_verified_lineage"]
    visual04 = sorted({r["frame_path"]: r for r in visual04}.values(), key=lambda r: float(r["selected_local_pts"]))
    frames = [(T4_FRAMES / r["frame_path"].replace("frames/", "")).resolve() for r in visual04]
    frames = [(p, float(r["selected_local_pts"])) for p, r in zip(frames, visual04) if p.is_file()]
    make_case_card("04", final_rows["04"], samples["04"], args.out / "q3_card_sample04.png", frames)
    make_case_card("03", final_rows["03"], samples["03"], args.out / "q3_card_sample03.png")
    make_case_card("15", final_rows["15"], samples["15"], args.out / "q3_card_sample15.png")
    make_case_card("13", final_rows["13"], samples["13"], args.out / "q3_sample13_anomaly.png", anomaly=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    values = [[abs(float(r[f"phi_cls_{m}"])) for r in final_rows.values()] for m in ("text", "audio", "vision")]
    axes[0].boxplot(values, tick_labels=["Text", "Audio", "Vision"], patch_artist=True,
                    boxprops={"facecolor": "#d9e6f2", "edgecolor": "#47789d"}, medianprops={"color": "#a33b3b", "linewidth": 1.5})
    axes[0].set_ylabel("Absolute classification Shapley contribution")
    axes[0].set_title("Magnitude distribution across 20 samples")
    counts = {m: sum(r["primary_reference_modality_cls"] == m for r in final_rows.values()) for m in ("text", "audio", "vision")}
    axes[1].bar(list(counts), list(counts.values()), color=["#3977A8", "#55a58a", "#d88c45"])
    axes[1].set_ylabel("Sample count (n=20)"); axes[1].set_title("Primary reference modality")
    axes[1].set_ylim(0, max(counts.values(), default=0) + 2)
    for i, v in enumerate(counts.values()): axes[1].text(i, v + .2, str(v), ha="center", fontsize=9)
    save(fig, args.out / "q3_modality_contribution_summary.png")

    metrics = read_json(args.results / "valid_metrics.json")
    matrix = np.asarray(metrics["confusion_matrix"]["rows_true_cols_pred"], dtype=int)
    fig, ax = plt.subplots(figsize=(5.6, 4.9), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), ["Negative", "Neutral", "Positive"]); ax.set_yticks(range(3), ["Negative", "Neutral", "Positive"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Valid split confusion matrix (n=728)")
    for i in range(3):
        for j in range(3): ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * .55 else "#1d2833", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=.82, label="Count")
    save(fig, args.out / "q3_valid_confusion_matrix.png")

    valid = read_csv(args.results / "valid_predictions.csv")
    color = {"Negative": "#4477AA", "Neutral": "#66A61E", "Positive": "#CC6677"}
    fig, ax = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    for label in ("Negative", "Neutral", "Positive"):
        selected = [r for r in valid if r["true_class"] == label]
        ax.scatter([float(r["true_intensity"]) for r in selected], [float(r["predicted_intensity"]) for r in selected], s=17, alpha=.58, c=color[label], label=label, edgecolors="none")
    lim = max(3.0, max(abs(float(r["true_intensity"])) for r in valid), max(abs(float(r["predicted_intensity"])) for r in valid))
    ax.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=1, linestyle="--", label="identity")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("True intensity"); ax.set_ylabel("Predicted intensity")
    ax.set_title("Valid split regression predictions (n=728)"); ax.legend(frameon=False, loc="upper left")
    save(fig, args.out / "q3_valid_regression_scatter.png")
    generated = sorted(p.name for p in args.out.glob("*.png"))
    expected = {"q3_card_sample04.png", "q3_card_sample03.png", "q3_card_sample15.png", "q3_sample13_anomaly.png", "q3_modality_contribution_summary.png", "q3_valid_confusion_matrix.png", "q3_valid_regression_scatter.png"}
    if set(generated) != expected:
        raise RuntimeError(f"figure output set mismatch: {generated}")
    print(json.dumps({"status": "FIGURES_COMPLETE", "figures": generated, "sample04_frames_included": len(frames)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
