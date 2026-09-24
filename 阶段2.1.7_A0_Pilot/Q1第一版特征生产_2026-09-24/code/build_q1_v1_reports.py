from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DELIVERY_ROOT = PROJECT_ROOT / "outputs" / "q1" / "v1_delivery"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    root = DELIVERY_ROOT
    rows = read_csv(root / "manifest.csv")
    validation = json.loads((root / "reports" / "automatic_validation.json").read_text(encoding="utf-8"))
    example_path = root / "examples" / "typical_correspondence_s9qJ7ATP7w_clip6.json"
    example = json.loads(example_path.read_text(encoding="utf-8"))
    modes = Counter(r["alignment_mode"] for r in rows)
    correspondence = Counter(r["text_audio_correspondence"] for r in rows)
    mapping_status = Counter(r["text_av_time_mapping_status"] for r in rows)
    speech_status = Counter(r["audio_speech_valid"] for r in rows)
    face_status = Counter(r["face_feature_valid"] for r in rows)

    numeric_fields = [
        "original_effective_duration_sec", "text_sequence_length", "audio_observation_length",
        "video_observation_length", "official_word_count", "valid_word_time_count",
        "audio_word_valid_count", "vision_word_valid_count",
    ]
    numeric = {}
    for field in numeric_fields:
        values = np.asarray([float(r[field]) for r in rows if r.get(field, "") != ""], dtype=np.float64)
        numeric[field] = {
            "count": int(values.size), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()), "sum": float(values.sum()),
        }
    total_zero = 0
    zero_samples = []
    word_level_gaps = []
    all_audio_windows = 0
    all_video_frames = 0
    for row in rows:
        archive = root / row["output_file"]
        with np.load(archive, allow_pickle=False) as z:
            zero = int(z["alignment_zero_duration_word_count"].item())
            total_zero += zero
            if zero:
                zero_samples.append({"sample_key": row["sample_key"], "zero_duration_word_count": zero})
            all_audio_windows += int(z["raw_audio_lld_values"].shape[0])
            all_video_frames += int(z["raw_video_pts_sec"].shape[0])
            for i in np.flatnonzero((z["word_time_valid"] == 1) & (z["word_audio_valid"] == 1) & (z["word_vision_valid"] == 0)):
                word_level_gaps.append({
                    "sample_key": row["sample_key"],
                    "word_index": int(i),
                    "official_word": str(z["words"][i]),
                    "word_start_sec": float(z["word_start_sec"][i]),
                    "word_end_sec": float(z["word_end_sec"][i]),
                    "audio_word_valid": int(z["word_audio_valid"][i]),
                    "vision_word_valid": int(z["word_vision_valid"][i]),
                    "video_observation_count_in_word": int(z["vision_word_feature_indptr"][i + 1] - z["vision_word_feature_indptr"][i]),
                    "handling": "word interval retained; video/face feature mask remains 0; no frame is fabricated",
                })

    with (root / "reports" / "word_level_observation_gaps.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(word_level_gaps[0]) if word_level_gaps else ["sample_key"])
        writer.writeheader()
        writer.writerows(word_level_gaps)

    def find(key: str) -> dict[str, str]:
        return next(r for r in rows if r["sample_key"] == key)

    silence = find("-mJ2ud6oKI8$_$1")
    mismatch = find("-mJ2ud6oKI8$_$6")
    no_face = find("-571d8cVauQ$_$0")
    zero_duration = find("-s9qJ7ATP7w$_$0")
    summary = {
        "schema_version": validation["schema_version"],
        "samples": len(rows),
        "feature_files": validation["feature_file_count"],
        "feature_bytes": validation["feature_bytes"],
        "execution_status": dict(Counter(r["status"] for r in rows)),
        "alignment_modes": dict(modes),
        "text_audio_correspondence": dict(correspondence),
        "text_av_time_mapping_status": dict(mapping_status),
        "audio_speech_valid": dict(speech_status),
        "face_feature_valid": dict(face_status),
        "total_native_audio_windows": all_audio_windows,
        "total_native_video_frames": all_video_frames,
        "native_sequence_statistics": numeric,
        "stable_ts_diagnostic_zero_duration_words": {"total": total_zero, "sample_count": len(zero_samples), "samples": zero_samples},
        "word_level_audio_valid_count": int(numeric["audio_word_valid_count"]["sum"]),
        "word_level_vision_valid_count": int(numeric["vision_word_valid_count"]["sum"]),
        "word_level_vision_masked_words": word_level_gaps,
        "a0_protection": json.loads((root / "reports" / "step06_full_output_audit.json").read_text(encoding="utf-8"))["checks"]["all_a0_baseline_hashes_match"],
        "typical_correspondence_sample": example,
        "representative_anomaly_rows": {"silence": silence, "mismatch": mismatch, "no_face": no_face, "zero_duration_diagnostic": zero_duration},
    }
    (root / "reports" / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mode_line = "、".join(f"{k} {v}条" for k, v in modes.items())
    method = f'''# Q1 第一版：方法与100条结果说明

版本：`q1-feature-v1.0`。本说明总结截至步骤07的实际产物；不包含情感分类模型训练或分类效果结论。

## 数据与处理范围

- 按官方 `label-100.xlsx` 的 `text` 列保留100条原文，不用ASR替代或改写；官方工作簿SHA-256：`{validation['label_xlsx_sha256']}`。
- 100条各有一个独立NPZ和manifest行，源MP4及输出NPZ均逐条记SHA-256；正式特征目录合计 `{validation['feature_bytes']:,}` bytes（约{validation['feature_bytes']/1_000_000:.2f} MB）。
- 每条原始音频和画面均从MP4按解码presentation PTS处理，使用共享呈现时间轴；音频重采样为16 kHz单声道，音视频原生观测保持各自真实序列长度。

## 三类特征

| 模态 | 实际方法 | 原生/派生维度 | 长度语义 |
|---|---|---:|---|
| 文本 | `FacebookAI/roberta-base` 固定revision；对官方词字符跨度内的subword最后隐层作均值 | 每词768维 | `text_sequence_length`为官方词数；100条共{int(numeric['official_word_count']['sum'])}词 |
| 音频 | PyAV按presentation PTS解码至16 kHz mono；openSMILE `eGeMAPSv02`低层描述量 | 原生25维；可信词级均值+总体标准差50维 | 原生窗长度独立记录；100条共{all_audio_windows:,}个音频窗 |
| 视觉 | PyAV解码帧presentation PTS；MediaPipe FaceLandmarker VIDEO提取blendshape | 原生52维；可信词级均值+总体标准差104维 | 帧长度与面部有效帧mask分别记录；100条共{all_video_frames:,}帧 |

## 词级与片段级对应规则

词时间使用稳定时间轴下的半开区间 `[start,end)`。只在对应关系证据支持词级使用时，按声学窗中心/视频帧PTS是否落入词区间选择观测，并保存索引；词级派生向量由所选观测计算均值和总体标准差（`ddof=0`）。未验证词映射的样本保留全部官方词特征及原生A/V序列，不制造词时间。

| 路由/关系结果 | 数量 | 处理含义 |
|---|---:|---|
| `TRI_MODAL_WORD_VALID` | {modes.get('TRI_MODAL_WORD_VALID',0)} | 已有内容对应证据的样本使用词级时间；其中视觉观测缺失的单词仍由逐词mask表示 |
| `AV_VALID_TEXT_UNALIGNED` | {modes.get('AV_VALID_TEXT_UNALIGNED',0)} | 保留官方文本和A/V序列，不将官方词映射到A/V时间 |
| `TRI_MODAL_WITH_AUDIO_CONTENT_INVALID` | {modes.get('TRI_MODAL_WITH_AUDIO_CONTENT_INVALID',0)} | 音轨存在但确认无有效人声；继续保留真实音频窗和原视频 |
| `UNCERTAIN_REVIEW` | {modes.get('UNCERTAIN_REVIEW',0)} | 机器证据不足以断言词级对应；当前第一版采用片段级文本关联及独立A/V时间序列，不作为逐条用户听辨任务 |

对应关系计数：`confirmed_match={correspondence.get('confirmed_match',0)}`、`confirmed_mismatch={correspondence.get('confirmed_mismatch',0)}`、`no_speech={correspondence.get('no_speech',0)}`、`not_asserted={correspondence.get('not_asserted',0)}`。`audio_speech_valid` 计数为 `{json.dumps(dict(speech_status),ensure_ascii=False)}`，其中 `-1` 表示没有足够证据判断，不等同于静音。历史全量审计的证据状态原样保留；本次生产无需把这些条目升级为人工通过，亦未向用户分派逐条复核。

## 100条覆盖与结构统计

- 处理状态：`{json.dumps(dict(Counter(r['status'] for r in rows)),ensure_ascii=False)}`；自动独立验证 `valid_count={validation['valid_count']}`、`failure_count={validation['failure_count']}`。
- 模态存在：text/audio/video 均为100/100；音频观测100/100。面部特征可用 `{face_status.get('1',0)}`/100；另有 `{face_status.get('0',0)}` 条无可用面部blendshape，但视频仍存在且帧PTS/掩码均保留。
- 官方词级有效时间合计 {int(numeric['valid_word_time_count']['sum'])} 个词；audio派生有效 {int(numeric['audio_word_valid_count']['sum'])} 个词，vision派生有效 {int(numeric['vision_word_valid_count']['sum'])} 个词。另有 `{len(word_level_gaps)}` 个已有词时间/音频派生但无可选有效视觉观测的位置，见 `word_level_observation_gaps.csv`；不填充虚构视觉值。
- effective duration：最短 {numeric['original_effective_duration_sec']['min']:.3f}s，中位数 {numeric['original_effective_duration_sec']['median']:.3f}s，最长 {numeric['original_effective_duration_sec']['max']:.3f}s。原生观测长度不是词数，也不应跨模态强行相等。
- stable-ts对齐诊断中共记录 {total_zero} 个零时长词，分布在 {len(zero_samples)} 条样本。这是诊断值；未通过内容对应路由的诊断时间不进入正式词级派生特征。所有样本的mapping-fail总计为0，但0 mapping-fail并不证明内容对应。
- 原A0 pilot及清单48项基准SHA全部一致，旧pilot NPZ/cache未被本次生产覆盖。

## 解释边界

这里的 `PASS` 表示源、schema、特征文件及对应规则通过工程结构检查，不表示100条官方文本都与人声相符，也不表示词边界达到毫秒级真值精度。仅5条有可用词级文本-A/V映射；其余条目仍完整保存可用模态和对应状态。`TRI_MODAL_WORD_VALID`样本中3个视觉词mask=0，已列入具体位置表。没有时间真值的语音内容、未知对应关系及无脸情况不作为缺失样本删除。
'''
    (root / "reports" / "Q1方法与结果说明.md").write_text(method, encoding="utf-8")

    words_md = []
    for item in example["word_rows"]:
        words_md.append(
            f"| {item['word']} | [{item['word_start_sec']:.3f}, {item['word_end_sec']:.3f}) | {item['audio_window_count']} | {item['video_frame_count']} | {item['illustration_frame_pts_sec']:.3f} |"
        )
    examples_md = f'''# 典型对应验证：`{example['sample_key']}`

## 样本及内容依据

- 官方原文：**{example['official_text']}**。
- 路由：`{example['sample_mode']}`；`text_audio_correspondence={example['text_audio_correspondence']}`；`text_av_time_mapping_status={example['text_av_time_mapping_status']}`。
- 内容证据：既有人工听辨记录确认，在应用edit list和忽略edit list的试听中均听到 **“But I just kept going”**；忽略edit list的版本前面还多出一句。这个证据支持所列短语存在，不等于证明整段音轨只含该短语。
- 源：`{example['source_relpath']}`，SHA-256 `{example['source_mp4_sha256']}`；NPZ SHA-256 `{example['feature_file_sha256']}`；stable-ts trace SHA-256 `{example['alignment_trace_sha256']}`。

## 共享呈现时间上的映射

解码音频覆盖 `[ {example['audio_decoded_presentation_start_sec']:.6f}, {example['audio_decoded_presentation_end_sec']:.6f} )` 秒；官方词区间最后结束于 `{example['word_rows'][-1]['word_end_sec']:.3f}` 秒，之后还有约 `{example['audio_after_last_official_word_interval_sec']:.3f}` 秒原始音轨。该尾部音频窗没有官方词区间，不据此推断具体语词，但仍完整保留在原生音频特征中。

| 官方词 | 词区间(s) | 被选音频窗数 | 被选视频帧数 | 示例帧PTS(s) |
|---|---:|---:|---:|---:|
{chr(10).join(words_md)}

音频和视觉观测索引均已从正式NPZ重新读取，并以 `center >= start and center < end` 重新计算；5个词的保存索引与独立重算结果逐词一致。示例图下排的5张图来自实际解码PTS帧，脸部区域为公开报告展示而马赛克处理；源MP4和特征文件未修改。

![共享时间轴上的音频波形、观测点和实际视频帧](typical_correspondence_s9qJ7ATP7w_clip6.png)

## 异常分支实例

| 分支 | 样本 | 保留与状态 |
|---|---|---|
| 确认静音 | `{silence['sample_key']}` | 官方文本仍编码；`audio_present=1`、`audio_speech_valid=0`；保留{silence['audio_observation_length']}个真实声学窗。该样本`video_present=1`，面部特征状态另由mask表示。 |
| 官方文本与人声错配 | `{mismatch['sample_key']}` | 官方文本仍是“{mismatch['official_text']}”；人工听辨未听到该句，原语音和视频保留；`text_av_time_mapping_status=unavailable`，不写入伪造词时间。 |
| 视频存在但无有效面部输出 | `{no_face['sample_key']}` | 保留{no_face['video_observation_length']}个视频帧的位置；`video_present=1`、`face_feature_valid=0`。面部值使用占位值且由mask标无效，不把视频写成不存在。 |
| 局部零时长词诊断 | `{zero_duration['sample_key']}` | 历史A0 trace中第4个`they`区间为`[0.78,0.78)`，结构上无有效持续时间；v1不把该时间作为可信词级映射，A0原件保持不变。 |

## 复核边界

本例支持“有内容证据的词区间能按共享PTS查询实际音频窗与视频帧”这一结构结论。它不测量精确边界误差，也不代表其他样本都可词级对齐；没有人工或强证据的内容对应没有被这张图升级为确认状态。
'''
    (root / "examples" / "典型对应验证.md").write_text(examples_md, encoding="utf-8")
    print(json.dumps({"summary": str(root / "reports" / "results_summary.json"), "method_report": str(root / "reports" / "Q1方法与结果说明.md"), "example_report": str(root / "examples" / "典型对应验证.md"), "word_level_masked_vision_words": len(word_level_gaps), "modes": dict(modes), "diagnostic_zero_duration_words": total_zero}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
