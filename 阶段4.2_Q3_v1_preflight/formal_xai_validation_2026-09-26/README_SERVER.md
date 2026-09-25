# Q3 解释验证运行包说明

本包用于已冻结 B0_seed2029 的附件2 valid 解释验证。完整执行范围是预冻结120条 valid 样本；正式输出始终停在 `REVIEW_GATE`，不自动使用 test 或附件4。模型、数据、scaler 和样本清单的 SHA-256 必须与 `assets/` 内的冻结记录相符。

## 环境和输入

- Python 3.12、numpy 2.5.0、torch 2.8.0；可使用经预检通过的 CPU 或 CUDA 环境。
- `aligned_50.pkl`：与冻结哈希一致的附件2数据。
- `run-dir`：已审计训练输出目录，其中有训练摘要、B0 三个种子的 checkpoint 与 scaler。
- `out`：新的空目录。输出不可覆盖或断点续写。

## 解包与核验

将 ZIP 放到受控工作区，并在隔离目录解包。先运行：

```cmd
python verify_xai_bundle.py <q3_xai_validation_bundle.zip>
```

同时对照外部 `q3_xai_validation_bundle.manifest.json` 中的 `archive_sha256`。运行包内的 `BUNDLE_MANIFEST.json` 记录所有源码文件 SHA；正式入口会再次验证这些字节。

## 执行

先在受控环境按实际路径填写参数，确认设备和输出目录：

```cmd
python run_formal_xai_validation.py --aligned-pkl <aligned_50.pkl绝对路径> --run-dir <训练输出目录绝对路径> --out <新空输出目录绝对路径> --device <cpu或cuda设备> --mode formal
```

`--mode smoke` 只运行冻结列表首样本，用于检查安装与入口，不能代替120条正式结果。正式模式固定运行120条；每条完成时保存逐样本 JSON 和进度。失败时保留 `failure.json` 及日志，回到审核门处理。

## 输出和解释边界

核对 `xai_validation_summary.json` 的 `sample_count=120`、数值失败列表、扰动区间和稳定性结果。审计文件包括 `run_identity.json`、`samples/`、`numeric_checks.json`、`sensitivity_checks.json`、`sensitivity_summary.json`、`reference_sensitivity.json`、`perturbation_summary.json` 和 `output_manifest.json`。

算法及参数见 `validation_contract.json` 和 `PROTOCOL_SOURCE.md`：八联盟精确 Shapley、固定原预测类、三档积分点数、三档替代比例、每档50个随机对照、video_id 分组 bootstrap 2000 次。弱归因或交互未定位时保留状态，不强制产生Top-k。valid上的文本、音频和视觉原素材映射均为 `index_only`，解释只能称为 feature-space attribution；不得宣称原词、音频秒数、视频帧、因果效应或“解释准确率”。数值 PASS 也不等于解释效果 PASS。
