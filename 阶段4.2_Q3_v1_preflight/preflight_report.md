# Q3 v1 preflight 报告（2026-09-25）

## 结论与阶段门

**受限 preflight 实现与 CPU smoke：PASS。Q3 4.2 原始证据映射硬门：OPEN。正式长时间训练/4.3：NO。**本报告只核验准备状态，不是 Q3 性能结果，更不是附件4预测与解释结果。

依据为 Notion《数学建模比赛 AI 协作执行协议》Q3 4.1/4.2（读取于 2026-09-25），其中 4.2 状态仍为 `PASS WITH REQUIRED FIXES`。协议允许接口实现、映射预检、CPU smoke 和归因单元测试；要求原始证据映射先于正式模型搜索关闭。此报告没有自行修改协议或将 Q3 标为 FROZEN。

## 1. Git 与隔离

- 发现旧本地克隆 `D:\Workspace\数学建模\shuxuejianmo-upload`：工作区干净，`main`，HEAD=`318ed399a5576ed58eba4b2da17783c5718f329d`，`origin` 为所给 GitHub URL。
- 远端 `main` 实查为 `5c525c28cd3eec410a2becc77f859a23e38e6816`。在可写工作区内由旧克隆建立独立克隆，更新 `origin` 至所给 GitHub URL，fetch 最新 `main`。
- 分支创建前再次核对干净工作区、`main`、HEAD 与 remote；随后从 `origin/main` 建立 `q3-v1-preflight`。不覆盖旧工作树，不改远端 `main`。

## 2. 实现与真实输入预检

| 项目 | 实际结果 |
|---|---|
| 附件2 `aligned_50.pkl` | 实算 SHA-256 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`，与仓库清单一致 |
| 数据合同 | train=3395、valid=728；两 split ID 不重叠；完整检查两 split 的三模态形状、非有限值、标签范围、BERT attention 前缀及 CLS/SEP；内容位置长度 train 1–48、valid 1–48 |
| 标准化 | 仅 train 有效内容位置拟合每维 mean/std，float64 统计、float32 网络输入；小标准差维数 text/audio/vision 均为 0；padding 重新置零，天然零行不剔除 |
| B0/B1 | 真实 train 样本 CPU 前向与反向通过，0 次优化器更新；B0 68,932 参数/275,728 FP32 权重字节，B1 125,380 参数/501,520 字节；输出范围与张量形状通过 |
| 预测接口 | `forward` 只接收 text/audio/vision/最终 validity mask；元数据变动不影响输出，非法额外输入被拒绝；B0/B1 padding 变动不影响输出 |
| Shapley/IG | 固定完整输入预测类别概率及单独回归目标；三参与者 8 组合精确 Shapley；条件 Gauss–Legendre IG；真实样本上未训练 B0 的 Shapley 残差 0、IG 64 步残差约 `-4.33e-08`，仅证明数值实现；合成交互例验证 `local_attribution_unresolved_interaction` |
| valid 解释子集 | 依预定 `SHA-256('q3-xai-v1|' + sample_id)` 规则确定前 120 条，清单 SHA-256 `aeaef63ae3906153a88bf58e9e3653edad3d2d32c95f11b9adf30c252d81991e`；未据此选模型或报告解释效果 |
| 单元测试 | 8/8 PASS；含模型、mask、接口隔离、train-only scaler、Shapley、IG 交互、映射 fail-closed、指标/训练门控 |

可复算机器详情见 `reports/preflight_machine_report.json`。未训练权重上的类别、模态排序没有情感解释价值，不据此作论文结论。

## 3. 原始证据映射硬门

预先固定附件4的 01、13、20 作为输入/证据预检样本，仅读取原始特征、文本与原视频文件哈希；**没有加载 Q3 预测模型对附件4推理**。三条样本的内容位置数分别为 22、30、43，合计 95 个序列位置、三模态共 285 条映射记录。`13` 的 aligned 视觉矩阵确为全零，原样保留。

**285/285 映射仍为 `index_only`，`verified_text=0`、`verified_time=0`。**目前只确认原始文件身份与官方序列位置；没有匹配并冻结 tokenizer revision，也没有可审计的官方 `text` 特征行来源规则；更没有证明官方 aligned 音频/视觉第 t 行来自原视频何时何帧。未生成虚构秒数、关键帧或“原词删除”结论。记录及输入 SHA 见 `reports/mapping_precheck.json`、`reports/representative_evidence_mapping.csv`。

独立质检确认：代码的受限 smoke 可通过，但 token CLS/SEP 结构不能单独证明三支路官方行的共同原始索引语义。**即使文本重分词 IDs 匹配，也不能自动证明预计算 text 矩阵行来源；文本时间对齐也不能代替音频/视觉官方行与原始区间的关系。**`verified_*` 仅在外部来源规则、人工内容核查、方法版本、映射表与哈希提供并复核后可使用。调用方布尔声明不是独立证据。

## 4. 已冻结范围与未做事项

- `frozen_config.json` 固定 aligned、预计算 text、B0/B1、双头 loss、AdamW、valid 选择分数、种子、均值参考、归因目标和典型样本选择规则。数值容差 `1e-6` 当前是代码预检候选；正式归因前仍需依重复性测试最终冻结。
- `q3v1/train_eval.py` 提供之后的 train/valid 评价与 checkpoint 接口，但训练入口要求独立映射门报告及其映射表、配置、复核报告哈希；本轮**没有调用它**。
- 没有正式训练、没有在 test 上比较模型或计算 test 指标、没有附件4正式推理、没有生成最终解释卡或性能表。由于官方 PKL 将 train/valid/test 放在单一 pickle 对象，反序列化会装载 test 字节；本轮代码从不索引 `container['test']` 或利用其结果。
- 本机为 Python 3.12.10、torch 2.8.0+cpu、numpy 2.5.0。服务器环境、CUDA/显存和完整 50 MB 全题附件预算尚需服务器及最终包检查。

## 5. 进入下一步的条件

当前 **READY_FOR_Q3_4_3=NO**。正式长时间 B0/B1 搜索前，应在不使用附件4预测结果调规则的前提下取得并核验：①预计算 `text` 与官方序列/原始文本的来源关系及匹配 tokenizer/offset；②官方 aligned 音频行和视觉行到原始时间/PTS 的各自支持关系；③多样本映射表、内容核查、适用范围、失败清单、方法版本和哈希。若某模态仍只能 `index_only`，该模态原始证据解释继续 STOP。随后再验证服务器环境并进行正式训练许可审查。

## 6. 复核记录

独立 P1 只读质检先发现缺少可复用训练/valid接口与文本映射证明门；已补入 `train_eval.py`，并要求全 50 token、attention、特殊位、来源文档 SHA 后才可标 `verified_text`。复核后 8 项测试通过，未发现阻塞**受限 preflight**的 P1 代码缺陷；原始行语义与原始证据门仍开放。测试日志、退出码随包保存。
