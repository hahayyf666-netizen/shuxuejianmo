# DR-X 服务器交接：先资格门，再 B0/B1

用户将在服务器自行执行。此页的命令应在 **Python 3.12 + `requirements-preflight.txt` 已安装**的隔离环境中运行。附件2路径请替换为服务器上的原件路径；服务器不需要 Git 或 GitHub。先把 `q3_scope_preflight_server_bundle.zip` 传到服务器，并把下载前后 SHA-256 与同目录的 `.manifest.json` 对比。

```bash
sha256sum q3_scope_preflight_server_bundle.zip
mkdir -p q3_scope_run
unzip -q q3_scope_preflight_server_bundle.zip -d q3_scope_run
cd q3_scope_run
python verify_bundle.py ../q3_scope_preflight_server_bundle.zip
python run_server_preflight.py \
  --aligned-pkl '/server/path/附件2-数据集特征文件/aligned_50.pkl' \
  --expected-hostname "$(hostname)" \
  --out reports/server_preflight
```

`run_server_preflight.py` 验证分模态合同与 C-4 哈希，执行9项单元测试，重验附件2 aligned SHA、train/valid 数量与隔离，做B0/B1零优化器步数值烟测，保存环境、stdout、stderr、退出码。成功时生成 `reports/server_preflight/q3_server_preflight_gate.json`，其 `status` 必须为 `PASS`。若任何步骤失败，保留日志并停止；不要手写 PASS 回执。若服务器 Python 包版本与冻结依赖不同，应先记录差异并复核影响，不能静默替换解释或预测算法。

正式训练命令在**服务器资格门 PASS 且报告经审核后**执行；本地本轮没有执行：

```bash
python run_formal_training.py \
  --aligned-pkl '/server/path/附件2-数据集特征文件/aligned_50.pkl' \
  --gate reports/server_preflight/q3_server_preflight_gate.json \
  --out runs/b0b1_v1 \
  --device cuda:0
```

如采用 CPU，可明确改为 `--device cpu`；设备由运行者根据服务器实际资源选择。正式训练入口拒绝非空输出目录，按冻结六个候选运行并保存checkpoint、逐轮valid历史、`training_summary.json`。在架构、checkpoint、scaler、解释协议和代码哈希锁定前，**不要用 test 比较候选**；附件4只在最终模型冻结后进行正式预测和特征空间解释。

向 Codex 回传：服务器 `hostname`、压缩包 SHA、`verify_bundle.py` 输出、`reports/server_preflight/` 全部日志与 JSON、CPU/GPU 信息、退出码。服务器资格门审核通过后再决定是否启动正式训练。报告中的 audio/vision `index_only` 始终不得输出为秒级或帧级原始证据。
