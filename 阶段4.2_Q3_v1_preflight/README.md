# Q3 v1 preflight（服务器运行冻结包）

状态：**Stage C-4 已审查；分模态解释范围已收口；DR-X 服务器资格门待运行，正式训练未开始。**以 `stage_d_explanation_scope_finalization/Q3_EXPLANATION_SCOPE_FINAL_REPORT.md` 和 `SERVER_HANDOFF.md` 为当前执行入口。本目录不是 Q3 最终模型、模型权重或附件4正式结果。

## 冻结边界

- 输入版本仅 `aligned_50`；预测文本表示仅附件2/4的预计算 `text`。`text_bert` 只用于内容位置结构检查，不进入预测网络。
- 预测 `forward(text_feat, audio_feat, vision_feat, validity_mask)`；`raw_text`、token ID、sample ID、原视频均留在解释支路。
- B0/B1、双头、训练配置、参考与解释规则见 `frozen_config.json`。Stage D 新增受服务器资格门保护的训练入口；模型、Shapley、conditional IG 仍保持原实现。
- 附件2 train 可拟合 scaler，valid 只用于结构预检；附件2 test 保持模型选择锁定。官方 PKL 是单一 pickle 容器，反序列化会装载 test 字节，但脚本不索引 test 分组或计算任何 test 指标。
- 附件4的 01、13、20 仅用于输入/证据映射预检，绝不加载 Q3 模型对其推理。`13` 全零视觉真实保留。
- 历史 preflight 的285个代表性索引当时均为 `index_only`。随后 C-4 仅对附件4 `01–20` 的文本内容行实证恢复 `564/564` 个 WordPiece/字符跨度；音频和视觉仍为 `index_only`。不得把其位置号换算为秒数、语音时段或关键帧。

## 服务器重跑（仅预检）

在本目录执行，先安装 `requirements-preflight.txt` 对应环境，并指定服务器上只读原始附件位置：

```bash
python -m unittest discover -s tests -v
python run_preflight.py --aligned-pkl '/path/to/附件2-数据集特征文件/aligned_50.pkl' --out reports
python run_mapping_precheck.py --aligned-directory '/path/to/附件4-可解释专项视频样本与特征文件/对齐版本' --out reports
```

输入文件应为赛题原件；附件2 aligned SHA-256 必须为 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`。`run_preflight.py` 会计算真实哈希并 fail closed。服务器须另记录 `hostname`、Python/Torch/CUDA、RAM、设备占用和命令退出码；本地 CPU 结果不能替代服务器资格门。

## 当前分模态解释范围与训练继续条件

正式解释范围采用 `TEXT_MAPPING_PASS_ATTACHMENT4_SCOPE`、`AUDIO_MAPPING_BLOCKED`、`VISION_MAPPING_BLOCKED`、`FEATURE_SPACE_ATTRIBUTION_ALLOWED`。附件4文本仅20条样本的内容行可回溯 WordPiece/字符跨度；音频和视觉只允许特征空间索引归因，不允许秒级、帧级、未经验证的通道物理语义或因果结论。正式长时间 B0/B1 训练须先由实际 DR-X 服务器运行 `run_server_preflight.py` 并取得可核验的 PASS 回执；当前未放行。

## 文件

- `q3v1/`：数据合同、B0/B1、训练/valid评价接口、Shapley、conditional IG、fail-closed evidence mapping 和分模态 scope gate。`fit_candidate` 仅在实际服务器 preflight PASS 后可调用。
- `stage_d_explanation_scope_finalization/`：最终解释合同、报告、服务器交接说明和新的冻结运行包；`run_server_preflight.py` 只做资格检查，`run_formal_training.py` 为之后的正式训练入口。
- `tests/`：不依赖原始数据的单元测试。
- `reports/`：真实数据预检、固定 valid 解释子集、代表性映射、日志及 scaler。
- `frozen_config.json`：原首轮模型与解释数学定义，Stage D 未修改。新的分模态可交付范围另见 `stage_d_explanation_scope_finalization/explanation_scope_contract.json`。
- `SOURCE_COMMIT.txt`、`SOURCE_MANIFEST_SHA256.txt`、`source_manifest.json`：包对应的已提交源码快照及其 SHA-256。
- `SHA256SUMS.json`、`q3_v1_preflight_server_bundle.zip`：历史 Stage A 包，保留用于审计；当前服务器使用 Stage D 的 `q3_scope_preflight_server_bundle.zip` 及其 manifest。

本包代码与报告供参赛队审核；使用前须按实际服务器环境重跑并核对原题、协议和映射证据。
