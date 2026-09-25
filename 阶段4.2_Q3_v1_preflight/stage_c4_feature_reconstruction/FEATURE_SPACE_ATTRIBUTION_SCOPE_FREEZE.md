# Q3 v1 feature-space attribution 技术备选范围（C-4 冻结）

状态：`TECHNICAL_FALLBACK_FROZEN / EXISTING_PROTOCOL_GATE_UNCHANGED`。本文件落实“严格音视频行级重建失败后的技术方案冻结”；**不是**对 Notion Q3 解释协议、`frozen_config.json` 或训练放行门的修订。

## 证据输入

- 附件4的 `raw_text → text_bert → text` 已在固定 BERT 权重下重建：564/564 内容行及1000/1000全部矩阵位置逐维最大误差≤`1e-4`，所以本批20条的文本映射可标 `verified_text`。字符区间对应 WordPiece，未必是完整自然语言词语。
- 原 MP4 → 官方 `audio` 74维行：`0/3677` 严格恢复，保留 `index_only`。
- 原 MP4 → 官方 `vision` 35维行：`0/2593` 严格恢复，保留 `index_only`。
- 候选特征序列的 CKA 不提升上述状态；C-2 的 aligned↔unaligned 行匹配也不等于原素材时间映射。

## 将来可执行的 feature-space attribution 输出合同

1. 以冻结的 aligned_50 三模态输入和原 B0/B1 数学定义为基础；分类、回归目标、训练/验证/test 隔离保持原合同。
2. 模态级可计算 exact 3-player Shapley，输出 text/audio/vision 对指定模型输出的相对贡献及基线、目标、残差。该值是**模型在所给特征上的响应**。
3. 局部可计算原定义的 conditional IG，输出 `sample_id`、`modality`、`official_seq_index`、`feature_channel_index`、归因值、基线、目标、completeness 残差与 `mapping_status`。统一名称为 `feature-space attribution`。
4. `mapping_status` 随每个可交付归因对象保存：本批附件4 text 内容行为 `verified_text`；audio/vision 在取得官方来源证明前为 `index_only`。其他样本不得从附件4结果自动继承 `verified_text`。
5. 可报告已有 XAI validation 对模型输出的稳定性/敏感性结果；不能称作解释准确率或真实情绪因果效应。

## 表述红线

在 audio/vision 官方行到原素材的支持位置未核验前，不把 seq_index 输出写成“关键语音片段/第几秒”“关键视频帧/表情事件”。不把音视频74/35维通道索引赋予未经生成记录验证的物理或表情含义。文本即使已可回到 WordPiece 字符区间，当前技术备选输出仍采用 feature-space attribution 名称；任何未来的自然语言“关键词证据”主张须另行核查归因目标、词片合并和展示规则。

## 执行门

`MAPPING_GATE = BLOCKED_FOR_RAW_AUDIO_AND_VISION`；`FEATURE_SPACE_TECHNICAL_SCOPE = FROZEN`；`Q3_EXISTING_PROTOCOL_GATE = STOP_UNCHANGED`；`B0_B1_FORMAL_TRAINING_RELEASE = NO`。若未来要以该范围进入正式训练，须另行修订并验收 Q3 协议；本次 C-4 不执行该修改。

来源：`Q3_FEATURE_RECONSTRUCTION_REPORT.md`、`results/text_row_trace_564.csv`、`results/media_reconstruction.json`、`results/c4_gate.json`，以及历史 C-1/C-2/C-3 审计。
