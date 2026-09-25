# Stage C-1：Evidence Mapping 根因诊断

日期：2026-09-25
范围：附件4固定代表样本 01、13、20；只诊断现有 285 行证据映射状态。未修改模型、冻结参数或预测接口；未进行正式训练、模型预测或附件4推理。

## 阶段结论

**TEXT_MAPPING：BLOCKED**

**AUDIO_MAPPING：BLOCKED**

**VISION_MAPPING：BLOCKED**

**ROOT_CAUSE：** 285 行全部为 index_only 的直接原因是 **B：原 precheck 路径对每个模态、每个位置无条件调用 index_only()**。因此 285/285 是同一 fail-closed 输出，**不是 285 次实际映射尝试失败**。同时存在 **A：当前 PKL 未提供逐行特征来源字段**；这使现有附件文件本身不足以证明特征行到原始内容的位置关系，但本轮没有证明该信息无法从上游特征提取代码、配置或来源记录恢复。

## 分模态判断

| 模态 | 状态 | 本轮证据与阻塞点 |
|---|---|---|
| Text | BLOCKED | 20/20 条附件4样本的 BERT tokenizer 重放与官方 text_bert 的 token IDs、attention mask、token type IDs 完全一致。固定代表样本 01/13/20 的 95 个内容位置均得到 raw_text 字符范围候选。但 Q3 输入使用的是 768 维连续 text 特征，PKL 不证明这些向量行由对应 token 生成；故不能把候选 offset 记为 verified_text。 |
| Audio | BLOCKED | 固定样本的原视频文件存在，PKL 中有 audio 矩阵，但未见逐行音频时间或特征支持区间。本轮没有运行 forced alignment，也没有核验官方特征行到原音轨区间的生成规则。仅从 raw video 重新得到候选词时间，仍不足以证明它就是官方 audio 特征行的来源。 |
| Vision | BLOCKED | 固定样本的原视频文件存在，PKL 中有 vision 矩阵，但未见逐行 PTS、帧号或特征支持信息。本轮没有核验官方 vision 行到原视频帧的生成规则。视频存在以及序列长度相同都不能单独证明逐行时间对应。 |

## 为什么会出现 285/285 index_only

旧预检脚本 run_mapping_precheck.py 在第 45 行附近对 text、audio、vision 的每个内容位置直接调用 index_only()；随后把全部行数作为 index_only 计数。该路径没有调用已经定义的 map_text_offsets() 或 map_av_support()，也没有产出候选映射失败类型。因此“全是 index_only”准确描述了预检输出策略，而不是模态映射的实测成功率。

固定样本的内容位置数为 22、30、43，合计 95。原映射表为每个位置记录三种模态，故共 95 × 3 = 285 行，每模态 95 行。原表状态为 index_only 285 行、verified_text 0 行、verified_time 0 行。

本轮另对附件4全部 20 个 PKL 做只读清点：观察到的顶层键为 audio、id、raw_text、text、text_bert、vision；未发现命名明显表示逐行 time、start/end、PTS、frame、timestamp 或 alignment 的字段。20/20 个 PKL 均有配对原视频。该结果说明这些逐行来源信息不在当前所检查的 PKL 字段中；它不证明上游提取流程或外部记录没有这些信息。

## Text 子映射结果

使用 google-bert/bert-base-uncased 固定 revision b96743c503420c0858ad23fca994e670844c6c05 的 vocab.txt，词表 SHA-256 为 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3。Python 3.12.10、Transformers 4.57.6 下，以 lowercase、max_length=50、truncation、padding=max_length 重放全部 20 条 raw_text，并导出 tokenizer offset mapping：

- 20/20 条的 token IDs、attention mask、token type IDs 与 text_bert 精确一致；
- 共获得 564 个内容 token 的字符范围；
- 在原 285 行中，text 子集的 95/95 个位置都有精确 raw_text → text_bert 字符 offset 候选；
- 连续 text 特征行来源仍为 0/95 已验证，所以 Text Mapping 整体仍为 BLOCKED。

该复现只使用 tokenizer 词表，没有加载 BERT 权重，没有生成预测。详细逐 token 结果见 text_tokenizer_replay_20.csv；固定 285 行及字符候选见 prior_285_root_cause.csv。

## A / B 分类与判断边界

- **B：直接原因，已确认。** 原预检代码路径无条件写入 index_only，未执行候选映射；所以之前的 285 行状态无法用于判断映射本身是否可做。
- **A：当前交付文件的证据缺口，已确认；全局不可恢复性，未确认。** 现有附件4 PKL 没有显式逐行来源字段。上游提取器、版本、配置、原始对齐记录是否可获得，以及能否重建并验证官方矩阵各行，尚未检查完成。
- 因而当前不能将任一模态标 PASS，也不能把 A 解释成“题目数据天然不允许对齐”。应先寻找与这批 PKL 对应的实际来源流水线和版本，再验证来源关系；若无法取得或复现该证据，就继续保持 BLOCKED / fail-closed。

公开文档可用于理解格式与候选工具，但不能替代本附件的来源证明。MMSA 数据说明将 text、text_bert、audio、vision 和长度作为主要字段；MMSA-FET 文档说明其可选 Wav2vec CTC aligner 可生成词时间并对齐音视频，但当前没有证据将该工具的特定版本和配置归属于比赛提供的附件4 PKL。[MMSA 数据格式说明](https://github.com/thuiar/MMSA/blob/master/README.md#2-datasets)；[MMSA-FET 对齐器说明](https://github.com/thuiar/MMSA-FET)。

## 复现记录

映射输入：原 preflight 的 reports/representative_evidence_mapping.csv；附件4对齐版本 01–20 的 PKL 与配对 MP4。每个样本的 PKL、MP4、raw_text SHA-256 见 attachment4_mapping_inventory_20.csv；原固定三样本的哈希也见原 reports/mapping_precheck.json。

执行的本地命令逻辑如下，参数路径按本机数据位置提供：

1. 使用固定 vocab、附件4对齐版本目录运行 probe_tokenizer_mapping.py，输出 attachment4_mapping_inventory_20.csv、text_tokenizer_replay_20.csv、tokenizer_replay_summary.json。
2. 运行 summarize_existing_285.py，将 tokenizer offset 候选与原 285 行逐位置合并，输出 prior_285_root_cause.csv、root_cause_summary.json。

两个脚本、配置锁定值、逐样本哈希和完整输出均随本目录提交。映射预检和归并脚本成功运行；初次调用系统默认 Python 时因未安装 NumPy 而未运行，随后切换至已有 Python 3.12.10 环境成功完成。该安装差异不改变审计数据或映射判断。

## 下一步

仅继续找回并核对这批 PKL 实际采用的特征生成来源：版本/提交、配置、文本 token 到连续 text 行的关系、音频行时间支持、视频行 PTS/帧支持及对应文件哈希。取得来源后，先在固定样本 01/13/20 上做候选重建和官方行支持核验；只有证据满足 frozen mapping contract 才可改映射状态。当前不启动 B0/B1 训练。
