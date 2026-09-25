# Q3 v1 DR-X CUDA 服务器资格门审核

审核日期：2026-09-26。审核对象为从 DR-X 下载的 `reports/server_preflight_cuda128/` 原始输出。本报告只审核资格门，不包含正式训练结果。

## 结论

**服务器资格门 PASS；允许按冻结的 B0/B1 六候选计划启动 train/valid 正式训练。**本次没有使用 test 选模型，也没有运行附件4正式推理。解释边界维持：附件4指定20条文本可回溯词片段；音频和视觉只允许特征空间索引归因，不得称为秒级语音或帧级视觉证据。

## 来源身份与环境

| 检查项 | 审核结果 |
|---|---|
| 服务器冻结包 SHA-256 | `79c07231aa08c7feb539956005e73ee5a2b60770f8dbdb7a6bed2375a34d3a4a`；服务器 `verify_bundle.py` 报告22个文件验证通过；与本地冻结包相同 |
| 冻结包 source commit | `cb66d39d30622f8649720312d0cc5146b7793ed7` |
| Scope contract SHA-256 | `303b4549f7954470e56c579a3befe8081ae3a670c5ef56f738e0606910e9f0bb`；与冻结包原件一致 |
| 数据原件 SHA-256 | `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`；与代码冻结值一致 |
| 环境 | `DR-X`，Python `3.12.14`，NumPy `2.5.0`，Torch `2.8.0+cu128`；`torch.cuda.is_available()=true` |
| GPU | 2 张 NVIDIA GeForce RTX 3090，各 25,769,279,488 字节显存；用户在 `cuda:0`、`cuda:1` 均完成了真实张量运算，结果各为 `2.0` |

## 回执与机器检查

- `q3_server_preflight_gate.json` 状态 `PASS`；text、audio、vision、explanation 四个分模态 Gate 与冻结合同一致，data、interface、xai、environment 均为 `PASS`。
- 9项单元测试全部通过；`unit_tests.exit_code.txt=0`，`data_smoke.exit_code.txt=0`，两份 stderr 未报告错误。数据烟测实际在 CPU 上运行，CUDA 可见性另由环境报告和两张卡的张量运算证明。
- 真实附件2载入后 `train=3395`、`valid=728`，样本 ID 交集为0；scaler仅用train拟合。B0/B1的参数量分别为68,932和125,380，输出为三类logits；损失有限、反向梯度存在、优化器步数均为0。
- Shapley分类和回归加和残差绝对值分别约 `8.7e-19` 和 `1.4e-17`；64步 IG 完备性残差约 `4.3e-8`，数值状态 `pass`。这些是未训练模型的数值烟测，不是解释效果指标。
- 回执中机器报告、环境报告和两个退出码文件的 SHA-256 均与下载文件一致；机器报告中 scaler、120条valid解释子集的 SHA-256 也均一致。公开文件的逐文件哈希另见 `SHA256SUMS.txt`。`train_scaler.npz` 已在本地核验，但二进制内容仅保留于本地服务器报告和本机下载目录，GitHub 同步包只记录其 SHA-256，不包含该文件。
- `test_used_for_model_selection=false`、`attachment4_model_inference_run=false`、`formal_training_run=false`；核对冻结的 `run_formal_training.py` 后，正式训练入口只读取 train/valid，并强制检查资格门、数据哈希、split隔离与空输出目录。

## 放行范围和下一步

可以在 DR-X 的 `SXJM/env` 环境启动冻结的 B0/B1、种子 `2029/2030/2031` 共六个候选；建议使用空闲的 `cuda:1`。选择只依据 valid。训练完成后先审核 `training_summary.json`、六条训练历史和 checkpoint 身份，再决定最终 test 及附件4步骤。

本次资格门没有在 GPU 上运行 B0/B1 整模型的前向/反向烟测。已确认两张 GPU 可完成张量运算；正式训练入口将首次运行整模型的 GPU 训练循环，若失败应保留日志并停止，不修改冻结结构或跳过 Gate。
