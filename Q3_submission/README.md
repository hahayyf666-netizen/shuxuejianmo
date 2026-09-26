# Q3 Submission Package

本目录是问题3（可解释性情感预测）的精简竞赛提交包。只保留题目要求的核心代码、说明文档、模型参数、配置、运行环境、附件4最终预测与解释结果，以及复核这些解释所需的最小证据文件。内部 T0–T4、Gate、审计日志和阶段性报告不作为最终参赛附件主体。

## 1. 目录结构

```text
Q3_submission/
├─ README.md
├─ ENVIRONMENT.md
├─ requirements.txt
├─ SUBMISSION_MANIFEST.json
├─ config/
│  ├─ q3_final_config.json
│  ├─ FINAL_EXPLANATION_SEMANTICS.json
│  └─ attachment4_explanation_contract.json
├─ code/
│  ├─ q3v1/
│  │  ├─ __init__.py
│  │  ├─ data.py
│  │  ├─ model.py
│  │  ├─ train_eval.py
│  │  └─ explain.py
│  ├─ run_training.py
│  ├─ run_attachment4_inference.py
│  ├─ run_valid_evaluation.py
│  └─ verify_against_submission.py
├─ model/
│  ├─ B0_seed2029.pt
│  └─ train_scaler.pt
├─ evidence/
│  ├─ text_row_trace_564.csv
│  ├─ t3_media_origin_records.json
│  ├─ t4_audio_evidence.csv
│  ├─ t4_visual_evidence.csv
│  ├─ final_evidence_coverage_20.csv
│  ├─ final_evidence_coverage_summary.json
│  ├─ attachment4_anomaly_summary.csv
│  └─ frames/
│     └─ T4核验关键帧
└─ results/
   ├─ attachment4_predictions_explanations.csv
   └─ valid_metrics_reference.json
```

## 2. 冻结模型与输入口径

- 特征版本：`aligned_50`。
- 模型：B0。文本/语音/视觉分别线性投影到64维，经 GELU 后按有效位置均值池化；三模态拼接后进入64维融合层，输出三分类 logits 与 `3*tanh` 回归强度。
- 架构选择：比较 B0/B1 在 seeds 2029/2030/2031 上的平均 valid 目标值 J。
- 交付 seed：2029，由冻结协议预先固定，并非从三个 seed 中按 valid 另选。
- scaler：只由附件2 train 拟合。
- 文本预测输入：预计算 `text`；`text_bert` 仅用于内容位置结构检查和文本证据映射。
- 分类解释：exact 3-player Shapley，目标固定为 full-input predicted-class probability。
- 回归解释：exact 3-player Shapley，目标为预测情感强度。
- 局部解释：conditional IG，仅移动一个模态，其余模态保持实际输入。

最终参数见 `config/q3_final_config.json`；解释术语以 `config/FINAL_EXPLANATION_SEMANTICS.json` 为准。

## 3. 环境

建议使用 Python 3.12：

```bash
pip install -r requirements.txt
```

已验证环境见 `ENVIRONMENT.md`。

## 4. 训练协议复现

准备赛题附件2 `aligned_50.pkl` 后运行：

```bash
python code/run_training.py \
  --aligned-pkl <附件2/aligned_50.pkl> \
  --out output/training \
  --device cuda:0
```

脚本只使用官方 train/valid；test 与附件4不参与模型选择。训练协议与冻结版本保持一致：B0/B1、三个固定 seed、CE+MAE、AdamW、early stopping、三 seed 平均 valid J 选架构。交付 seed 2029 按冻结协议固定。

内部开发阶段的服务器 Gate 不属于科学模型本身，因此不放入竞赛提交包；本脚本是提交用的便携训练入口。`code/q3v1/train_eval.py` 仅保留训练/评估所需公共函数，不包含内部 Gate 依赖。

## 5. Valid 指标复现

```bash
python code/run_valid_evaluation.py \
  --aligned-pkl <附件2/aligned_50.pkl> \
  --checkpoint model/B0_seed2029.pt \
  --scaler model/train_scaler.pt \
  --out output/valid_metrics.json \
  --device cpu
```

冻结参考值：

| Metric | Value |
|---|---:|
| Accuracy | 0.6442307692 |
| Macro-F1 | 0.6070737874 |
| Negative F1 | 0.6502463054 |
| Neutral F1 | 0.4444444444 |
| Positive F1 | 0.7265306122 |
| MAE | 0.5964004835 |
| RMSE | 0.8137669799 |
| Pearson | 0.6290740125 |

## 6. 附件4正式推理

附件4 aligned 特征目录应恰好包含 `01.pkl` 至 `20.pkl`。

```bash
python code/run_attachment4_inference.py \
  --attachment-dir <附件4_aligned_50目录> \
  --checkpoint model/B0_seed2029.pt \
  --scaler model/train_scaler.pt \
  --contract config/attachment4_explanation_contract.json \
  --text-trace evidence/text_row_trace_564.csv \
  --t3-records evidence/t3_media_origin_records.json \
  --out output/attachment4 \
  --device cpu
```

脚本生成20条样本的极性、概率、强度、Shapley、conditional IG、top位置以及数值检查，并会核对冻结 checkpoint/scaler 与20个附件4 aligned PKL 的输入哈希。模型推理阶段 Audio/Vision 局部位置默认仍为 `index_only`；经过独立证据链核验的语音时段与视觉关键帧保存在 `evidence/` 中，并已经整合进最终提交 CSV。

## 7. 与最终 CSV 核对

```bash
python code/verify_against_submission.py \
  --generated output/attachment4 \
  --submitted results/attachment4_predictions_explanations.csv
```

核对内容包括：
- predicted class；
- 三类概率；
- predicted intensity；
- 分类与回归的三模态 Shapley；
- 六组 top sequence positions。

数值容差为 `1e-6`。原素材证据字段属于已冻结的证据映射注释，不由核心模型重新生成。

## 8. 最终附件4结果解释

`results/attachment4_predictions_explanations.csv` 共20条：

- `primary_reference_modality_cls`：对当前预测类别概率具有最大正向 Shapley 的模态。
- `dominant_influence_modality_cls`：分类 Shapley 绝对值最大的模态，其符号可能支持也可能反对预测。
- `dominant_influence_modality_reg`：回归 Shapley 绝对值最大的模态。
- conditional IG 的 top positions 按绝对重要性排序，同时保留 signed importance；top-|IG| 不能自动解释为“正向支持”。
- 原素材覆盖 20/20、20/20、19/20 的含义是：相应参考模态的所列重要位置中至少一处具有已验证的原素材证据，不表示所有 top positions 都完成映射。
- Sample04：Audio 2/7、Vision 3/7 个已列位置具有验证证据。
- Sample05 Vision 仍为 `feature_position_only`。
- Sample13/16 的视觉异常仅解释为 feature-branch attribution，不声称存在已验证原始关键帧。
- Sample06/18 保留视觉来源链歧义边界。
- Shapley / conditional IG 是模型特征空间归因，不是因果情绪贡献。

## 9. 提交说明

题目要求问题2/3提交核心代码、说明文档、模型参数文件、配置文件和运行环境，并提交附件4预测与解释 CSV；整个竞赛附件总大小不超过50 MB，且严禁出现参赛单位、队员姓名、队伍编号等身份信息。

本 Q3 包不包含：
- 赛题原始 PKL / MP4；
- 内部审计日志、Gate、T0–T4全过程材料；
- 临时输出、缓存、中间模型；
- 论文正文中的图片副本。

附件4正式结果文件：
`results/attachment4_predictions_explanations.csv`
