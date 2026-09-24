from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SOURCE_NPZ = PROJECT_ROOT / "outputs/q1/v1_delivery/features/-s9qJ7ATP7w___6.npz"
STEM = "q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_font() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei"):
        if name in available:
            return name
    return "DejaVu Sans"


def within_dimension_zscore(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Standardize every feature dimension across words; constant dimensions become zero."""
    matrix = np.asarray(values, dtype=np.float64).T  # feature x word
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, ddof=0, keepdims=True)
    constant = std[:, 0] < 1e-12
    safe_std = std.copy()
    safe_std[constant] = 1.0
    z = (matrix - mean) / safe_std
    z[constant] = 0.0
    return z, int(constant.sum())


def write_long_csv(
    path: Path,
    modalities: list[tuple[str, np.ndarray, np.ndarray, list[str]]],
    words: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "modality",
                "feature_index",
                "feature_name",
                "word_index",
                "word",
                "word_start_sec",
                "word_end_sec",
                "raw_value",
                "within_feature_zscore_across_words",
            ]
        )
        for modality, raw_word_by_feature, z_feature_by_word, names in modalities:
            for feature_index, feature_name in enumerate(names):
                for word_index, word in enumerate(words.tolist()):
                    writer.writerow(
                        [
                            modality,
                            feature_index,
                            feature_name,
                            word_index,
                            word,
                            f"{float(starts[word_index]):.8f}",
                            f"{float(ends[word_index]):.8f}",
                            f"{float(raw_word_by_feature[word_index, feature_index]):.9g}",
                            f"{float(z_feature_by_word[feature_index, word_index]):.9g}",
                        ]
                    )


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_NPZ, allow_pickle=False) as archive:
        sample_key = str(archive["sample_key"].item())
        official_text = str(archive["official_text"].item())
        alignment_mode = str(archive["alignment_mode"].item())
        mapping_status = str(archive["text_av_time_mapping_status"].item())
        words = archive["words"].astype(str)
        starts = archive["word_start_sec"].astype(np.float64)
        ends = archive["word_end_sec"].astype(np.float64)
        text_raw = archive["text_word_feat"].astype(np.float64)
        audio_raw = archive["word_audio_feat"].astype(np.float64)
        vision_raw = archive["word_vision_feat"].astype(np.float64)
        text_valid = archive["text_word_valid"].astype(np.uint8)
        audio_valid = archive["word_audio_valid"].astype(np.uint8)
        vision_valid = archive["word_vision_valid"].astype(np.uint8)
        audio_lld_names = archive["raw_audio_lld_feature_names"].astype(str).tolist()
        blendshape_names = archive["raw_video_blendshape_names"].astype(str).tolist()

    if alignment_mode != "TRI_MODAL_WORD_VALID" or mapping_status != "word_valid":
        raise RuntimeError("The selected sample is not a trusted word-level tri-modal sample.")
    if not (np.all(text_valid == 1) and np.all(audio_valid == 1) and np.all(vision_valid == 1)):
        raise RuntimeError("The selected illustration requires all three word-level masks to be valid.")
    if not (
        text_raw.shape == (len(words), 768)
        and audio_raw.shape == (len(words), 50)
        and vision_raw.shape == (len(words), 104)
    ):
        raise RuntimeError("Unexpected formal feature dimensions.")

    text_z, text_constant = within_dimension_zscore(text_raw)
    audio_z, audio_constant = within_dimension_zscore(audio_raw)
    vision_z, vision_constant = within_dimension_zscore(vision_raw)

    text_names = [f"roberta_hidden_{index:03d}" for index in range(text_raw.shape[1])]
    audio_names = [f"mean:{name}" for name in audio_lld_names] + [
        f"std:{name}" for name in audio_lld_names
    ]
    vision_names = [f"mean:{name}" for name in blendshape_names] + [
        f"std:{name}" for name in blendshape_names
    ]
    modalities = [
        ("text", text_raw, text_z, text_names),
        ("audio", audio_raw, audio_z, audio_names),
        ("vision", vision_raw, vision_z, vision_names),
    ]

    data_csv = HERE / f"{STEM}_data.csv"
    write_long_csv(data_csv, modalities, words, starts, ends)

    font_name = choose_font()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    width_in = 180 / 25.4
    height_in = 155 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), layout="constrained")
    grid = fig.add_gridspec(4, 1, height_ratios=[0.28, 1.25, 1.0, 1.0])
    header = fig.add_subplot(grid[0, 0])
    header.axis("off")
    header.text(
        0.5,
        0.78,
        "词级三模态特征热图：But I just kept going",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    header.text(
        0.5,
        0.17,
        "颜色仅表示各特征维度在本句5个词之间的相对变化；不表示情感强度，也不用于跨模态原值比较。",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#333333",
    )
    panels = [
        ("A  文本特征", text_z, "RoBERTa隐藏维度（768）"),
        ("B  语音特征", audio_z, "eGeMAPS统计维度（50）"),
        ("C  视觉特征", vision_z, "面部blendshape统计维度（104）"),
    ]
    axes = []
    image = None
    tick_labels = [
        f"{word}\n[{start:.2f}, {end:.2f}) s"
        for word, start, end in zip(words.tolist(), starts.tolist(), ends.tolist())
    ]
    for panel_index, (title, values, ylabel) in enumerate(panels):
        ax = fig.add_subplot(grid[panel_index + 1, 0])
        image = ax.imshow(
            values,
            cmap="RdBu_r",
            vmin=-2.0,
            vmax=2.0,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            rasterized=True,
        )
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=4)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xticks(np.arange(len(words)))
        ax.set_xlim(-0.5, len(words) - 0.5)
        ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, len(words), 1), minor=True)
        ax.grid(which="minor", axis="x", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False)
        if panel_index < 2:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlabel("官方词及其共享presentation-time半开区间", fontsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#4D4D4D")
        axes.append(ax)

    if image is None:
        raise RuntimeError("No heatmap was created.")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.88, pad=0.02)
    colorbar.set_label("同一特征维度在5个词间的标准分数 z", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    outputs = {
        "png": HERE / f"{STEM}.png",
        "svg": HERE / f"{STEM}.svg",
        "pdf": HERE / f"{STEM}.pdf",
    }
    fig.savefig(outputs["png"], dpi=600, facecolor="white", transparent=False)
    fig.savefig(outputs["svg"], facecolor="white", transparent=False)
    fig.savefig(outputs["pdf"], dpi=600, facecolor="white", transparent=False)
    plt.close(fig)

    with Image.open(outputs["png"]) as rendered:
        png_info = {
            "width_px": rendered.width,
            "height_px": rendered.height,
            "mode": rendered.mode,
            "dpi": list(rendered.info.get("dpi", ())),
        }

    checks = {
        "selected_sample_is_word_valid": alignment_mode == "TRI_MODAL_WORD_VALID",
        "mapping_status_is_word_valid": mapping_status == "word_valid",
        "all_text_words_valid": bool(np.all(text_valid == 1)),
        "all_audio_words_valid": bool(np.all(audio_valid == 1)),
        "all_vision_words_valid": bool(np.all(vision_valid == 1)),
        "word_intervals_finite_monotone": bool(
            np.all(np.isfinite(starts))
            and np.all(np.isfinite(ends))
            and np.all(ends > starts)
            and np.all(starts[1:] >= ends[:-1] - 1e-12)
        ),
        "all_transformed_values_finite": bool(
            np.all(np.isfinite(text_z))
            and np.all(np.isfinite(audio_z))
            and np.all(np.isfinite(vision_z))
        ),
        "formal_dimensions_match": bool(
            text_raw.shape[1] == 768 and audio_raw.shape[1] == 50 and vision_raw.shape[1] == 104
        ),
    }
    manifest = {
        "purpose": "Auxiliary manuscript validation of word-position feature organization; not a performance or emotion-intensity result",
        "sample_key": sample_key,
        "official_text": official_text,
        "alignment_mode": alignment_mode,
        "text_av_time_mapping_status": mapping_status,
        "source_npz": SOURCE_NPZ.relative_to(PROJECT_ROOT).as_posix(),
        "source_npz_sha256": sha256(SOURCE_NPZ),
        "transformation": {
            "formula": "z[d,w] = (x[d,w] - mean_w x[d,w]) / std_w x[d,w], population std (ddof=0)",
            "constant_dimension_policy": "z=0 when std across the five words is below 1e-12",
            "selection": "all formal dimensions retained; no feature selection",
            "color_limits": [-2.0, 2.0],
            "interpretation_boundary": "within-dimension relative variation only; no cross-modality raw-value comparison and no emotion-intensity meaning",
        },
        "dimensions": {"text": list(text_raw.shape), "audio": list(audio_raw.shape), "vision": list(vision_raw.shape)},
        "constant_dimensions": {"text": text_constant, "audio": audio_constant, "vision": vision_constant},
        "word_intervals_sec": [
            {"word": str(word), "start": float(start), "end": float(end)}
            for word, start, end in zip(words, starts, ends)
        ],
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "font": font_name,
        },
        "outputs": {
            name: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
        "data_csv": {
            "path": data_csv.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": data_csv.stat().st_size,
            "sha256": sha256(data_csv),
            "rows": int((768 + 50 + 104) * len(words)),
        },
        "png_metadata": png_info,
    }
    manifest_path = HERE / f"{STEM}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not manifest["all_checks_pass"]:
        raise RuntimeError("Figure audit failed; see manifest.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
