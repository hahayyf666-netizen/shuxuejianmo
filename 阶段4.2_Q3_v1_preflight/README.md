# Q3 v1 preflight（服务器运行冻结包）

状态：**实现与 CPU 预检通过；原始证据映射硬门未通过；不得启动正式长时间训练。**本目录不是 Q3 最终模型、模型权重或附件4正式结果。

## 冻结边界

- 输入版本仅 `aligned_50`；预测文本表示仅附件2/4的预计算 `text`。`text_bert` 只用于内容位置结构检查，不进入预测网络。
- 预测 `forward(text_feat, audio_feat, vision_feat, validity_mask)`；`raw_text`、token ID、sample ID、原视频均留在解释支路。
- B0/B1、双头、训练配置、参考与解释规则见 `frozen_config.json`。本包只允许 preflight，不含训练入口。
- 附件2 train 可拟合 scaler，valid 只用于结构预检；附件2 test 保持模型选择锁定。官方 PKL 是单一 pickle 容器，反序列化会装载 test 字节，但脚本不索引 test 分组或计算任何 test 指标。
- 附件4的 01、13、20 仅用于输入/证据映射预检，绝不加载 Q3 模型对其推理。`13` 全零视觉真实保留。
- 285 个代表性 `official_seq_index` 当前全为 `index_only`。不得把位置号换算为秒数、语音时段或关键帧；不得用文本时间对齐自动证明官方音频/视觉行的原始支持区间。

## 服务器重跑（仅预检）

在本目录执行，先安装 `requirements-preflight.txt` 对应环境，并指定服务器上只读原始附件位置：

```bash
python -m unittest discover -s tests -v
python run_preflight.py --aligned-pkl '/path/to/附件2-数据集特征文件/aligned_50.pkl' --out reports
python run_mapping_precheck.py --aligned-directory '/path/to/附件4-可解释专项视频样本与特征文件/对齐版本' --out reports
```

输入文件应为赛题原件；附件2 aligned SHA-256 必须为 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`。`run_preflight.py` 会计算真实哈希并 fail closed。服务器须另记录 `hostname`、Python/Torch/CUDA、RAM、设备占用和命令退出码；本地 CPU 结果不能替代服务器资格门。

## 映射硬门继续条件

正式长时间 B0/B1 搜索前，必须另外提供预冻结的 tokenizer revision 与可复核的官方 text 行来源关系；分别证明官方 aligned 音频行、视觉行与原始素材时间/帧支持区间的关系，保存多样本映射表、人工内容核查、失败清单、方法版本和哈希。若只能证明文本位置，音频/视觉原始证据继续 STOP。映射不能通过时可保留序列级结果，但不能声称完成题目要求的原始音视频局部证据。

## 文件

- `q3v1/`：数据合同、B0/B1、冻结的训练/valid评价接口、Shapley、conditional IG、fail-closed evidence mapping。`fit_candidate` 在映射门未通过时直接拒绝执行，预检脚本不调用它。
- `tests/`：不依赖原始数据的单元测试。
- `reports/`：真实数据预检、固定 valid 解释子集、代表性映射、日志及 scaler。
- `frozen_config.json`：拟议首轮模型与解释规则。数值容差仍是预检候选，须在正式解释前完成重复性冻结。
- `SOURCE_COMMIT.txt`、`SOURCE_MANIFEST_SHA256.txt`、`source_manifest.json`：包对应的已提交源码快照及其 SHA-256。
- `SHA256SUMS.json`、`q3_v1_preflight_server_bundle.zip`：由 `freeze_bundle.py` 从上述 Git commit 内容生成的包内清单与归档；`python verify_bundle.py q3_v1_preflight_server_bundle.zip` 核对所有条目。

本包代码与报告供参赛队审核；使用前须按实际服务器环境重跑并核对原题、协议和映射证据。
