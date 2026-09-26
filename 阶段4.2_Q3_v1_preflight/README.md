# Q3 v1（冻结技术分支）

> 本 README 只用于记录 Q3 技术分支的**当前冻结状态、交付边界与证据入口**。它不是最终竞赛 submission README。Stage A–D / preflight / T0–T4 的历史文件继续保留用于审计，但其中的阶段性限制不能当作当前交付状态。

## 当前状态（2026-09-26）

- 冻结模型：`B0_seed2029`。B0/B1 通过三种子平均 valid 目标值选择架构；交付种子 2029 由冻结协议预先固定，并非从三个种子中按 valid 另选。
- 没有重新训练、修改网络、损失、阈值、Shapley 或 conditional IG。
- valid120 XAI 数值验证与预定扰动检查已完成；解释结论限于模型特征空间响应，不作因果解释。
- valid728、test727 与附件4 20条正式预测/解释均已完成；附件4无标签，不报告准确率。
- T4-aware finalization candidate 位于 `q3_finalization_candidate_2026-09-26/`，其中包含最终派生 CSV、valid 错误分析、test 一致性分析、7张论文候选图以及冻结结果复现记录。
- 外部 4.6 复核最终结论：`PASS_WITH_LIMITATIONS`，当前 P0=0、P1=0；限制均已明确披露，不再阻止进入 4.7。
- 历史 `q3_4_6_gate.json` 保留原样，表示当时阶段状态，不作为当前 4.6 结论。
- 4.7 尚未在本分支执行；下一步应创建独立的最终 submission package，而不是继续扩展模型/T4/XAI 审计。

DR-X 服务器资格门与训练证据见 `server_preflight_drx_cuda128_2026-09-26/` 和 `formal_training_audit_drx_cuda128_2026-09-26/`。冻结 test CSV、附件4逐样本结果、checkpoint 与 scaler 可在 `external_review_handoff_2026-09-26/artifacts_extracted/` 阅读。原始约 1 GB 的 `aligned_50.pkl` 未放入公共仓库，端到端重跑仍需赛题原始文件。

## 当前冻结技术口径

- 输入版本：`aligned_50`。
- 预测网络只使用预计算 `text / audio / vision` 和 `validity_mask`；`raw_text`、token ID、sample ID、原视频不进入预测 forward。
- 文本预测输入为预计算 `text`；`text_bert` 只用于内容位置结构检查与文本证据映射。
- scaler 仅由 train 拟合；valid 用于架构/训练过程选择；test 不参与模型、阈值或解释规则选择。
- 分类输出为三类情感极性；回归输出为 `3*tanh` 范围内的情感强度。
- 模态级解释使用 exact 3-player Shapley；分类“主要参考模态”定义为对 full-input predicted-class probability **最大正向 Shapley** 的模态。
- “主导影响模态”定义为最大 `|Shapley|`，其符号可能支持也可能反对当前预测。
- 局部位置使用 conditional IG；top positions 按绝对重要性排序，但保留 signed importance，不能把所有 top-|IG| 位置都称为正向支持。

完整最终配置见：
`q3_finalization_candidate_2026-09-26/q3_final_config.json`

完整解释语义见：
`q3_finalization_candidate_2026-09-26/FINAL_EXPLANATION_SEMANTICS.json`

## 当前原素材证据边界

- 文本：附件4 20条样本的内容行可回溯到已验证 WordPiece / 字符跨度。
- 音频与视觉：历史默认状态为 `index_only`；T4 **只对通过来源链和媒体定位核验的具体重要位置**提供可回看的语音时段或重建关键帧。
- 未通过 T4 的其余音频/视觉位置仍为 `feature_position_only` / `index_only`，不能按序列索引比例推算秒数或帧。
- 当前原素材证据覆盖：
  - classification primary reference：20/20；
  - classification dominant influence：20/20；
  - regression dominant influence：19/20。
- 上述覆盖率只表示相关模态所列重要位置中**至少一处**具有已验证原素材证据，不表示该模态所有重要位置均已映射。
- Sample04：Audio 2/7、Vision 3/7 个已列位置具有验证证据。
- Sample05 Vision 仍为 `feature_position_only`。
- Sample13/16 的视觉异常只允许解释为冻结预处理下的 feature-branch attribution，不声称存在已验证原始关键帧。
- Sample06/18 保留视觉来源链歧义边界。
- T4 状态保持 `T4_COMPLETE_WITH_LIMITATIONS`。

逐样本计数和证据状态见：
`q3_finalization_candidate_2026-09-26/results/final_evidence_coverage_20.csv`

## 当前结果与交付候选入口

优先阅读以下当前文件，而不是早期 preflight 状态文件：

- `q3_finalization_candidate_2026-09-26/README.md`：finalization candidate 说明。
- `q3_finalization_candidate_2026-09-26/q3_final_config.json`：冻结最终配置。
- `q3_finalization_candidate_2026-09-26/FINAL_EXPLANATION_SEMANTICS.json`：最终解释术语和边界。
- `q3_finalization_candidate_2026-09-26/results/attachment4_predictions_explanations_final.csv`：附件4 20条最终派生预测与解释表。
- `q3_finalization_candidate_2026-09-26/results/final_evidence_coverage_20.csv`：逐样本原素材证据覆盖。
- `q3_finalization_candidate_2026-09-26/results/valid_metrics.json` 与 `valid_error_analysis.json`：valid728 性能与错误归因。
- `q3_finalization_candidate_2026-09-26/results/test_consistency_summary.json`：test727 双头一致性分析，仅作描述，不用于调参。
- `q3_finalization_candidate_2026-09-26/results/reproduction_check.json`：valid/test/Attachment4 冻结结果复现记录。
- `q3_finalization_candidate_2026-09-26/figures/`：论文候选解释卡与性能图。
- `external_review_handoff_2026-09-26/`：外部审核材料与公开冻结工件。

## 已确认限制

当前 4.6 的 `PASS_WITH_LIMITATIONS` 保留下列客观边界：

1. Sample05 Vision 没有完成原素材映射，仍为 `feature_position_only`。
2. Audio/Vision 只有部分重要位置拥有经验证的原素材回看证据。
3. Sample13/16 视觉异常不能被解释为真实视频关键帧贡献。
4. 历史 exact invocation provenance 仍为 partial；当前复现只证明冻结实现可以复现冻结输出。
5. Shapley / conditional IG 是 feature-space attribution，不是因果情绪贡献，也不是“解释准确率”。

这些限制已进入当前语义与结果文件，不要求继续扩展 T4，也不要求修改模型或重训。

## 历史 preflight / Stage A–D 文件

本目录中的 `reports/`、`stage_d_explanation_scope_finalization/`、`formal_xai_validation_2026-09-26/`、`formal_xai_review_2026-09-26/`、T0–T4 以及其他 Gate/manifest 文件均保留用于审计。

其中一些历史文档会出现类似：

- “附件4 01/13/20 仅用于输入或映射预检”；
- “音频/视觉均为 index_only”；
- “test/附件4尚未正式推理”；
- “仍等待后续阶段门”。

这些文字描述的是**当时阶段**，不再代表当前冻结状态。特别是 01/13/20 的“仅预检”约束只适用于早期 preflight 运行；正式冻结后附件4 01–20 已全部完成预测与解释。

如需复核早期服务器 preflight，可继续使用历史脚本：

```bash
python -m unittest discover -s tests -v
python run_preflight.py --aligned-pkl '/path/to/附件2-数据集特征文件/aligned_50.pkl' --out reports
python run_mapping_precheck.py --aligned-directory '/path/to/附件4-可解释专项视频样本与特征文件/对齐版本' --out reports
```

这些命令仅用于重现历史资格/映射预检，**不是最终 submission 的运行说明**。

## 4.7 边界

4.7 应新建独立、精简的 submission package，并提供专门的 submission README。最终提交包应包含题目要求的核心代码、冻结模型/参数、配置、环境与运行说明、附件4最终预测解释结果和必要论文图；不应把大量内部 T0–T4、Gate、审计历史材料作为最终参赛交付主体。
