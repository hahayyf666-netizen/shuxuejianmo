# Q3 explanation scope finalization 报告

日期：2026-09-25。依据：已独立审查的 Stage C-1/C-2/C-3/C-4。C 阶段来源搜索至此终止。本次只修订解释范围与阶段门，并准备服务器资格检查和 B0/B1 训练入口；没有训练、test 指标读取、附件4正式预测或推理。

## 1. 原协议与 C-4 证据

原 Notion 4.2 把三模态原始证据逐行映射作为长时间 B0/B1 训练的统一硬门。C-4 实际从原始文本、音频和画面重建候选并与官方附件4 `01–20` 特征逐行比较：文本 `564/564` 内容行按预设逐维 `≤1e-4` 容差恢复，`20/20` 的 `text_bert` ID/attention/segment 与原文本 tokenizer 重放相等；音频 `0/3677`、视觉 `0/2593` 个官方完整行获得严格验证。音频25D候选与官方74D、视觉52D候选与官方35D维度不同，诊断用 CKA 不能证明官方行的时间窗或PTS。零行已验证不等于证明将来无法恢复。

## 2. 最终分模态解释合同

| 模态 | Gate | 可交付位置与表述 | 仍被禁止的表述 |
|---|---|---|---|
| Text | `TEXT_MAPPING_PASS_ATTACHMENT4_SCOPE` | **仅附件4 `01–20`** 的内容行标 `verified_text`，可回溯 WordPiece 与原始字符跨度；特殊符号、padding及附件2其他样本不自动继承 | 把WordPiece直接称为完整自然语言词；未核查的样本也标 `verified_text` |
| Audio | `AUDIO_MAPPING_BLOCKED` | `index_only`，可输出官方序列位置/通道的 feature-space attribution | 秒级片段、声音事件或未经核验的74维通道物理语义 |
| Vision | `VISION_MAPPING_BLOCKED` | `index_only`，可输出官方序列位置/通道的 feature-space attribution | 视频帧、PTS、表情事件或未经核验的35维通道物理语义 |

`FEATURE_SPACE_ATTRIBUTION_ALLOWED` 允许对**模型输入特征**报告 exact 3-player Shapley 模态贡献、原定义的 conditional IG 局部归因及其数值验证。数值归因只刻画冻结模型对给定特征的响应，不能称为真实情绪因果贡献、解释准确率，也不能把音视频索引包装成原始证据。文本展示若引用原句，须带 `sample_id`、官方位置、字符跨度、source SHA 和映射状态。

本协议修订属于对已知来源限制的诚实降级。题目对局部关键证据的原始素材回溯要求仍需在论文中逐模态说明实际可达边界；不得声称音频/视觉原始证据已经闭合，更不得称 Q3 全部任务已经完成。

## 3. 模型与执行边界

- `q3v1/model.py`、`q3v1/explain.py`、`frozen_config.json` 均未修改；SHA-256 分别为 `a9a2d3d047a3920771eeecf411c8e679d6b5b5ed5295b3439023040330308963`、`593ef464898909011b78a401b4eeee4b25fa25e3c5ea249d1abc35b2ea59a08f`、`9e73a6d40641e29718eec7b859179790030fe1462c94a56bed0cc10468ec7ecc`。新合同启动时逐项核哈希。
- 只改 `q3v1/train_eval.py` 的放行检查：从“三模态原始映射统一 PASS”改为“分模态解释合同核验 + 实际服务器 preflight PASS”；loss、优化器、早停、valid 选择指标和模型定义不变。
- `run_server_preflight.py` 依次运行单元测试与真实附件2 train/valid CPU 数值烟测，记录主机/Python/Torch/CUDA、stdout/stderr/退出码和源 SHA；**不做优化器更新**。只有该服务器回执通过，`fit_candidate` 才解除训练入口阻断。
- `run_formal_training.py` 是未来训练入口：B0/B1 各按 `2029/2030/2031` 三种子运行，只由 valid 选择架构；test 不参与选择，附件4不用于调参。该入口本轮没有执行。

## 4. Gate 判定与剩余条件

| Gate | 本轮状态 |
|---|---|
| `TEXT_MAPPING_PASS_ATTACHMENT4_SCOPE` | PASS，限制为20条与564个内容行 |
| `AUDIO_MAPPING_BLOCKED` | 保持 BLOCKED_FOR_RAW_EVIDENCE |
| `VISION_MAPPING_BLOCKED` | 保持 BLOCKED_FOR_RAW_EVIDENCE |
| `FEATURE_SPACE_ATTRIBUTION_ALLOWED` | PASS，仅特征空间归因 |
| 本地代码和合同单元测试 | 9/9 PASS；本机 `Finesse` 上的完整资格脚本干运行也返回PASS，正式训练0次、CUDA不可用；不等于DR-X服务器资格通过 |
| 服务器环境、数据、接口和XAI资格门 | **PENDING；用户将在DR-X自行运行** |
| B0/B1正式训练 | **尚未开始；服务器回执 PASS 后入口才可运行** |

详见 `explanation_scope_contract.json`、`SERVER_HANDOFF.md`、`q3v1/scope_gate.py` 和 C-4 `results/c4_gate.json`。历史 C-1/C-2/C-3/C-4 工件保持原样，不再进行来源搜索。

Notion《数学建模比赛 AI 协作执行协议》4.2 的范围、映射Gate及服务器资格门已按上述状态定点更新并回读验证：[协议页面](https://app.notion.com/p/3e3bcbf1217a8093b353fc7a76dd9b26)。历史 C-4 报告中的“当时协议仍STOP”保留为当时事实，不回写历史结论。
