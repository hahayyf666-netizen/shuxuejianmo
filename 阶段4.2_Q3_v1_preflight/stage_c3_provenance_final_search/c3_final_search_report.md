# Stage C-3：特征生成来源链最终检索与 Q3 解释范围决策

日期：2026-09-25
基线提交：`f65d3f9d2f56372fc7c9ca82699ab7a2b8ae632c`（`q3-v1-preflight`）

## 决策摘要

**在当前竞赛附件、本地仓库和公开上游材料中，没有找到能与本批附件 PKL 逐项对应的官方特征生成记录。**未找到附件4特征生成脚本/配置、特征模型与权重哈希、逐行对齐日志，或 audio 行到原音频窗口、vision 行到原视频 PTS/帧的清单。检索结论仅表示“已检查来源中未找到”，不推断上游绝对没有保存这些记录。

因此，当前最终可达边界是：

- **可确认**：附件4 aligned 与 unaligned 的样本身份及部分内部特征行关系；`raw_text → text_bert` 的 tokenizer 重放关系；原 MP4 音视频流存在可解码、有限且单调的 PTS。
- **不可确认**：连续 text 特征行由哪个 token/编码器层生成；audio 特征行支持哪个原音频时间窗；vision 特征行支持哪个原视频 PTS/帧。
- **当前 Q3 阶段门**：`TEXT_MAPPING=BLOCKED`、`AUDIO_MAPPING=BLOCKED`、`VISION_MAPPING=BLOCKED`；`READY_FOR_Q3_4_3=NO`。按现有 Notion 4.2.10/4.2.12，正式长时间训练、test评价、附件4正式推理与解释仍不放行。不能把 index-only 结果记为原始证据映射闭合。

## 本轮检索范围与证据

### 1. 本地竞赛数据与题目材料

- 递归盘点 `D:/Workspace/数学建模/E题数据`：文件类型为 140 个 MP4、102 个 PKL、2 个 XLSX 和 1 个 `.DS_Store`；未发现 `.py/.json/.yaml/.yml/.toml/.ini/.cfg/.log/.md/.txt/.csv/.tsv/.xml` 来源、配置或日志文件。
- 附件4目录只有对齐版/未对齐版各 20 个 PKL 及配套 40 个 MP4，无脚本、配置、提取运行记录或逐行时间/帧清单。具体配对 SHA 与跨版本行匹配已由 Stage C-2 保存。
- 赛题原始 DOCX SHA-256：`38cec978723bff36ab945fa98f6454558bb9084e2cd01d814dd6e38ab4b4c336`。DOCX 说明附件4沿用附件2字段、维度及时间组织，但未给出提取器版本、参数、模型资产、每行采样规则或逐行媒体支持位置。处理后的 PKL 不含原始波形、视频帧或词时间戳。详见 `../stage_c2_provenance/c2_provenance_report.md`。

### 2. 仓库与历史

- 检查 `q3-v1-preflight`、本地可见分支和仓库历史；可见代码包含 Q1 特征生产和 Q3 预检/映射代码，没有生成竞赛附件2或附件4 PKL 的来源程序、环境锁、日志或行级 provenance manifest。
- 仓库的 `dataset_manifest.json` 只证明附件2大型 PKL 的下载分片重组及整文件 SHA-256，不记录其特征提取过程；仓库 Git 历史中的附件4文件也没有生成日志或来源清单。

### 3. 公开上游材料核对

| 公开材料 | 其直接内容 | 与竞赛附件的核验 | 能否当成本批来源证明 |
|---|---|---|---|
| [thuiar/MMSA README](https://github.com/thuiar/MMSA) | 发布标准 MOSEI aligned/unaligned PKL 的 SHA-256、字段 schema，并说明预计算文本特征来自 BERT | 本地附件2 aligned SHA-256 为 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`，unaligned 为 `77eda14a06be9749a96c52ae45470c7cffcfa7219011eae391d231a0664c3762`；与 README 公布的 `45eccf...`、`ad8b23...` 不同 | 否。没有字节级身份；差异不能证明或排除使用相同工具/派生流程 |
| [thuiar/Self-MM README](https://github.com/thuiar/Self-MM) | 列出早期 MOSEI PKL 的 SHA-1 | 本地附件2 aligned SHA-1 为 `7629ba9eb1492bb1a4e5ae96d25114c84769a564`，unaligned 为 `344d7b87f478fbbb2c1d344be560f9e2d7b105da`；与 README 公布的 `ef4958...`、`db3e2c...` 不同 | 否。只是排除了与这两个已公布文件的逐字节相同 |
| [thuiar/MMSA-FET](https://github.com/thuiar/MMSA-FET) | 一个通用多模态特征提取工具仓库 | 没有找到声明该仓库某提交、配置或模型资产生成了竞赛附件2/4这些确切文件的链接、manifest 或 hash 对照 | 否。工具能力和维度相似性不建立实际文件来源链 |
| [CMU-MultimodalSDK MOSEI source definition](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py) 与 [process_mosei.py](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/examples/mmdatasdk_examples/full_examples/process_mosei.py) | 官方 SDK 定义 COVAREP、OpenFace2、FACET 4.2 等计算序列；示例脚本把 high-level 序列按 GloVe 词序列聚合/填补，再按标签对齐并生成长度50的 tensors。SDK 的 CSD 容器保存 feature intervals 和 metadata | 可作为标准 CMU-MOSEI 上游流程的候选说明；题目附件将 `text` 描述为 BERT 类特征，本地附件2哈希也不等于 MMSA/Self-MM 公布包。未找到该 SDK 的 CSD/提交/配置与本批 PKL 的精确对应或逐行向量匹配 | 否。它说明某标准流程及其能力，不证明竞赛附件实际由该流程产生 |

公开 MMSA 文档中 MOSEI 的 768/74/35 维 schema 与附件结构相似，但附件2整体哈希不相同，且附件4是单样本 PKL。相似字段/维度最多说明格式或数据族相近，不能据此把通用工具文档宣称为本批附件的官方生成记录。也未找到比赛组织方发布的附件2/4 特征生成提交、精确版本、模型权重 SHA、配置或运行记录。

CMU SDK 的标准 CSD 格式本身包含特征区间及元数据，理论上可作为未来恢复时间映射的候选原始来源；但必须先取得与竞赛特征数值逐行匹配且完整性信息可核验的对应计算序列。当前没有这种匹配结果。CMU SDK 官方仓库的历史 issue 曾报告 MOSEI CSD 下载链接不可用；本轮对两个官方 CSD URL 的浏览器访问也返回“当前浏览工具无法访问”，这不能证明源服务器本身已经关闭。本轮没有下载官方 CSD，也没有把 Hugging Face 上标注为 unofficial 的镜像作为来源证明。故不把“存在标准上游 interval CSD”误写成“本批特征行已经找到时间戳”。

## 最终 mapping 可达边界

| 映射链 | 已证实 | 未证实 | 最终状态 |
|---|---|---|---|
| `raw_text → text_bert` | 固定词表 tokenizer 对附件4 20/20 条的 IDs、attention、segment 重放一致；共 564 个内容位置得到字符 offset 候选 | 这不能证明连续 `text` 向量每行由相同 token、编码器层或权重生成 | tokenizer 子关系 PASS；`TEXT_MAPPING=BLOCKED` |
| aligned `text` 行 → token | 对齐/未对齐版本 20/20 样本完整 text 矩阵一致 | 实际编码器、权重、层、行生成规则与特征行—token 关系；18/20 的 `text` 在 attention padding 行仍有非零值，不能按非零模式定位 | BLOCKED |
| aligned audio 行 → unaligned audio 行 | 564/564 个内容行有唯一逐维精确匹配 | 每行在原 MP4 音轨的实际时间窗、窗口聚合/采样支持 | 内部行关系 PASS；`AUDIO_MAPPING=BLOCKED` |
| aligned vision 行 → unaligned vision 行 | 523/564 个内容行有唯一逐维精确匹配；41 个未匹配行在样本 13（30）和 16（11） | 每行对应的原视频帧、PTS、抽帧/人脸选择/聚合规则；41 行例外的生成原因 | 部分内部行关系 PASS；`VISION_MAPPING=BLOCKED` |
| 原 MP4 PTS | 20/20 个原视频音轨和视频流均有有限、非递减解码 PTS | PKL 中任何特征行与这些 PTS/音频窗口的关联 | 媒体时钟元数据 PASS；行到媒体时间 mapping BLOCKED |

上述数值复核来自 Stage C-2：独立校验脚本对原始 PKL 重查 1128 条音频/视觉行记录，PASS；未验证任何音/视频特征行的真实时间/帧支持。样本13、16的 vision 异常不据此解释为无画面或无脸。

## Q3 解释输出范围决策

[Notion《数学建模AI》当前 Q3 协议](https://app.notion.com/p/3e3bcbf1217a8093b353fc7a76dd9b26) 的 4.2.10/4.2.12 明确要求：原始证据映射是正式训练和附件4解释前的硬门；index-only 是中间状态，不满足原始证据要求；不能核验的模态保留 STOP。4.4 又要求局部证据可回溯。因此本次决策为：

1. **按当前协议，Q3 继续保持 STOP，不启动正式 B0/B1 长训练、不评价 test、不对附件4预测或生成解释卡。**这保留了当前冻结合同，不擅自降低阶段门。
2. **如果后续由参赛者正式修订协议并接受降级范围，技术上诚实的最大解释范围只能是模型输入空间：**训练并冻结模型之后报告验证集/附件4上的分类与回归 Shapley 模态贡献；报告 IG 或重要性到官方 `seq_index`、模态和特征通道索引，并标记 `mapping_status=index_only`。它们只说明模型对所给特征张量的响应。
3. 降级范围**不得**写成词/短语证据，不得给音频时间戳、视频帧/PTS、表情事件归因，不得给 74/35 特征通道赋予未经来源文件证明的具体物理/情绪语义，也不得声称局部贡献已回溯到原素材。Shapley/IG 是模型相对参考输入的归因，不是因果效应，也不是解释准确率。
4. 该降级范围不满足当前协议“可回溯局部关键证据”的要求，不能标记 Q3 解释闭合或 Q3 FROZEN。**当前正式可交付仍没有模型解释结果；模型空间归因只是未来经协议修订后可用的上限。**

## 后续解锁所需的来源材料

若要恢复原始证据级解释，至少需要与确切附件 hash 对应的生成流水线提交/版本、配置、text 编码器及权重 SHA、音频窗口起止/采样规则、视觉帧 PTS/抽帧规则、对齐表或足以逐行复现官方向量的完整环境与输入。视觉还需解释样本13/16的 41 条未匹配行。获得这些材料后先独立核验来源，再重新评估 mapping gate；在此之前不从平均帧率、有效长度或文本 forced alignment 推造官方特征时间。

## 检索结论

`OFFICIAL_GENERATION_RECORD_FOUND_FOR_EXACT_ATTACHMENT_HASHES = NO
`TEXT_MAPPING = BLOCKED
`AUDIO_MAPPING = BLOCKED
`VISION_MAPPING = BLOCKED
`READY_FOR_Q3_4_3 = NO
`TRAINING_RUN = NO
`ATTACHMENT4_INFERENCE = NO
