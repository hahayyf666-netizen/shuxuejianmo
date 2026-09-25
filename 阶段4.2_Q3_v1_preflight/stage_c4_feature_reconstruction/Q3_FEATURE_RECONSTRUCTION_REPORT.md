# Stage C-4：附件4原始模态到官方特征的最后一次重建验证

日期：2026-09-25。范围：附件4样本 `01–20` 的原始 `raw_text`、原 MP4 音轨/画面及配对的 aligned/unaligned 官方 PKL。本轮只做特征来源诊断；Q3 预测模型、损失、Shapley、conditional IG、Notion 协议和历史 C-1/C-2/C-3 工件均未修改，也未训练、评价 test 或推理附件4。

## 判定口径

“严格行级恢复”要求从原始输入经固定候选提取器得到与官方行**相同维度、相同样本与位置**的向量，并逐维检查。文本因 CPU 浮点实现差异预设 `max_abs_error ≤ 1e-4`；768 维每个坐标都须满足，且用样本 `01` 选候选，`02–20` 为不参与选择的复核样本。对音频和视觉，若候选通道数与官方不同或仅有表示相似度，则记作“未严格验证”，不把代理行写成官方时间/帧映射。

音视频另报告线性 CKA：先对各通道在片段内标准化，再比较两组按**诊断用** 20 Hz/15 Hz 网格组织的序列 Gram 矩阵。CKA 可在通道数不同的矩阵间计算；它既不是逐行向量相似度，也不能证明提取器、时间窗、视频帧或特征语义相同。倒序候选序列的 CKA 仅作时间顺序对照。没有以官方特征拟合投影、归一化参数或对齐偏移。

## 分模态结果

| 模态 | 从原始输入生成的候选 | reconstructed-to-official similarity | 严格行级恢复率 | 失败原因或边界 |
|---|---|---|---:|---|
| Text | `raw_text` → 固定 BERT tokenizer → BERT-base-uncased 最后一层，使用官方 attention mask | 564 个内容行的平均逐维 MAE `6.45×10⁻⁷`、最大绝对误差 `3.79×10⁻⁵`；逐行余弦约 `1.0000000000` | **564/564 = 100%**；全部 1000 个矩阵位置也满足阈值 | 附件4这20条的连续 `text` 行可回到对应 WordPiece/字符范围；这不自动外推到附件2所有样本，也不使音视频映射成立 |
| Audio | 原 MP4 presentation 音轨 → 16 kHz → openSMILE eGeMAPSv02 的 25 维 LLD | 官方 74 维与候选 25 维不能算同维逐行余弦/MAE；诊断用线性 CKA 在20条上的中位数 `0.3447`，倒序对照 `0.1091` | **0/3677 = 0% 已验证** | 没有重建官方 74 维提取器和每行原音轨支持窗；20 Hz 网格只是近似诊断 |
| Vision | 原 MP4 解码 PTS → 候选15 Hz帧 → MediaPipe 52 维 blendshape | 官方 35 维与候选 52 维不能算同维逐行余弦/MAE；16条可比较样本的线性 CKA 中位数 `0.5377`，倒序对照 `0.1954` | **0/2593 = 0% 已验证** | 没有重建官方 35 维提取器、脸选择和每行 PTS/帧支持；另4条没有足够有效候选脸行计算 CKA |

上述 `0%` 的含义是**本次没有得到严格证实的官方行**，并非证明原始媒体和官方特征毫无关系，更不证明上游永远无法复现。CKA 正序在所有可比较样本中高于倒序（audio `20/20`、vision `16/16`），提示存在共同时间结构；它仍不能替代官方行到原素材的来源验证。

### Text 的实证链

1. 沿用 C-1 已加固的 tokenizer 验证，先从每条 `raw_text` 重放 ID、attention、segment，再与 PKL 的 `text_bert` 比较；20/20 完全相等。词表 SHA-256 为 `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`。
2. 加载 [google-bert/bert-base-uncased 固定提交](https://huggingface.co/google-bert/bert-base-uncased/tree/8229d58a8e9c4f761cdb4a3f0434f856e1ae1d5d) 的 `model.safetensors`，实测 SHA-256 为 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。该资产未上传 GitHub；复现时按固定提交下载并核哈希。
3. 在样本 `01` 预定比较 13 个 hidden-state 层 × 官方/全1两种 attention mask；最小 MAE 为“官方 mask + 第12层” `6.19×10⁻⁷`。同样第12层但全1 mask 的 MAE 为 `0.14875`，官方 mask + 第11层为 `0.27549`。固定胜出配置后，样本 `02–20` 的全部内容行同样满足 `1e-4`。
4. 逐行明细见 `results/text_row_trace_564.csv`，包含样本 ID、官方序列位置、token ID、原文本字符区间、官方/重建行 SHA、余弦、MAE 和最大误差。20×50=1000 个位置都重建成功；特殊符号和 padding 位置不能称为“词语证据”。

数值结果将 C-3 的“Text 未证实”推进为**仅针对附件4这20条的 `verified_text`**。它证明原始文本到当前官方连续 text 行的功能性重建，不宣称取得了比赛组织方的原始运行日志或代码提交。

### Audio 与 Vision 的实测边界

- 两版 PKL、视频身份均按 C-2 的 SHA 清单在运行时重新核对；没有从近似采样率生成正式 `verified_time`。音频候选每条提取成功，20 Hz 网格覆盖官方有效行，额外用第一官方通道作为**未经证实的 F0 假设**与 openSMILE F0 代理比较：20条 Pearson 中位数约 `0.465`，波动为 `-0.190…0.739`。该单通道相关性不能确认其物理含义或恢复全部74维。
- 视觉候选使用真实解码 PTS 选最近视频帧，20条全部提取成功；2593 个官方有效行都有候选帧，MediaPipe 在其中2251个候选位置检测到脸。样本 `01`、`13`、`20` 没有可用候选脸；样本 `16` 仅有一个官方视觉有效行，均不能计算有意义的 CKA。候选脸覆盖与官方35维特征有效性不等价。
- 官方 74/35 维的确切生成代码、通道定义、模型资产、采样/聚合/标准化配置和行级音频窗/帧清单仍未找到。公开的 [MMSA-FET](https://github.com/thuiar/MMSA-FET) 只是候选工具资料；[CMU-MultimodalSDK 的 MOSEI 来源定义](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py) 也不能代替本批附件的生成证明。

## Gate 与解释范围决定

| 判定项 | 本轮状态 |
|---|---|
| 附件4 `text` 内容行 → 原文本 token/字符区间 | `verified_text`，564/564 |
| 附件4 audio 行 → 原音频时间窗 | `index_only`，严格验证 0/3677 |
| 附件4 vision 行 → 原视频 PTS/帧 | `index_only`，严格验证 0/2593 |
| 完整三模态原始证据映射 | **未闭合** |
| 现行 Q3 协议下正式 B0/B1 训练放行 | **NO，原阶段门保持 STOP** |

按照用户本轮“若严格行级验证失败则冻结”指令，已形成独立的 `FEATURE_SPACE_ATTRIBUTION_SCOPE_FREEZE.md` 技术备选范围：未来归因只命名为 **feature-space attribution**，保留每模态 mapping 状态；不声称音频秒级、视频帧/表情事件或真实因果贡献。该技术范围不覆盖、也不修改现行 Notion Q3 解释协议及其训练门。Shapley、conditional IG、B0/B1 的定义和代码没有变化。本轮没有生成任何 Q3 预测或解释结果。

## 复现与文件

- 文本：`reconstruct_text.py`；输入附件4 aligned `01–20.pkl`、固定 vocab、固定 BERT 权重；输出 `results/text_reconstruction.json` 与 `results/text_row_trace_564.csv`。
- 音视频：`reconstruct_media.py`；输入附件4 unaligned `01–20.pkl` 与同编号原 MP4，先按 C-2 清单验 SHA；复用 Q1 仅用于候选抽取的解码/openSMILE/MediaPipe 实现；输出 `results/media_reconstruction.json` 和20个候选 NPZ。`analyze_similarity_controls.py` 产生倒序对照；`summarize_c4.py` 核对样本集合、来源 SHA、564行阈值和候选工件 SHA，并输出 `results/reconstruction_summary_20.csv`、`results/c4_gate.json`。
- 运行环境及包版本见 `results/c4_gate.json`。候选脚本和数据使用 Python 3.12.10；文本模型仅做前向编码，不加载 Q3 预测器。
- BERT 440 MB 权重保存在工作区的 `work/q3_reconstruction_assets/hf_cache`，不进入 Git。逐个样本的输入 SHA 见 C-2 `pair_inventory_20.csv` 与本轮 JSON；候选 NPZ 的 SHA 见 `media_reconstruction.json`。
- 本目录 `README.md` 提供逐命令复现入口；`INDEPENDENT_READONLY_REVIEW.md` 记录独立只读复核。独立复核样本 `20` 的文本最大误差约 `4.36×10⁻⁵`，与本次主运行的 `3.79×10⁻⁵` 略有浮点差异，均低于预设 `1e-4`；仅 `3/564` 内容行在四位小数舍入后全部768维相等，不应表述为逐位复现。

### 四项阶段汇报

① **完成：**20条原始文本、音频、视频的候选重建，逐行文本核验和音视频代理相似度/倒序对照，保存代码、命令日志、逐样本表及机器 gate 回执。② **结论：**文本564/564行严格恢复；音频、视觉没有严格恢复任何官方完整行。③ **未确认：**官方74维声学与35维视觉提取来源及其每行时间/帧支持；本轮文本结论是否可外推到附件2全量。④ **阶段条件：**技术备选解释范围已单独冻结；现行 Q3 协议的原始证据门仍未满足，本轮不放行正式训练。
