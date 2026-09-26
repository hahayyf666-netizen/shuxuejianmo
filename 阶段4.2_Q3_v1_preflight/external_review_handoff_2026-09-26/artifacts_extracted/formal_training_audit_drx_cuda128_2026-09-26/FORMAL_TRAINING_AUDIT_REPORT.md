# Q3 v1 正式训练产物审核（DR-X / CUDA 12.8）

审核日期：2026-09-26。范围仅为附件2 train/valid 的 B0/B1 六候选训练、选择结果和检查点；本轮未执行 test、附件4正式推理或解释结果生产。

## 结论

**TRAINING_ARTIFACT_AUDIT = PASS。**六次候选均有完整逐轮 valid 历史，六个检查点均可按冻结模型严格载入，保存的轮次及指标与历史一致。根据冻结选择式 (J=0.5(1-\mathrm{MacroF1})+0.5\mathrm{MAE}/6)，B0 三种子的平均最优 (J=0.2528953740)，B1 为 (0.2579906923)；B0 较低，差值约 (0.0050953182)，大于冻结平局阈值 (10^{-4})。因此架构选择 B0，交付种子按预先冻结的 `2029`，对应 `B0_seed2029.pt`。

这表示**模型选择与训练产物可以锁定**，不表示 Q3 整题已完成。test 和附件4仍处于后续阶段门之外。

## 输入身份与审核方法

- 用户从 DR-X 下载的归档 `b0b1_v1_cuda128_audit.zip`：2,212,924 字节，SHA-256 `a945699a1ab8308cf51a3c92e95ccf680e9312d050bd2ceb293278959a8141e7`；ZIP CRC 校验通过，包含预期的 11 个文件，无多余文件。
- 训练回执引用的服务器资格门、解释合同和附件2原件哈希，均与此前冻结和已公开的审核材料一致。回执设备为 `cuda:1`，stderr 为空，进程退出码由用户在服务器确认是 `0`。
- `audit_training_bundle.py` 对六条历史、每条最优 (J)、检查点中的 variant/seed/epoch/valid 指标、状态字典形状、参数量及权重有限性逐项检查；`audit_result.json` 的 15 项检查全部通过。
- `replay_selected_valid.py` 使用本机同一 SHA-256 的附件2原件，只访问 train/valid，重新从 train 拟合 scaler，使用受限 `weights_only` 载入服务器 scaler，并逐项比较三模态均值与标准差；六个数组最大绝对差均为 `0`。随后载入所选检查点，在 CPU 上重算 728 条 valid 指标。官方 PKL 容器反序列化时会装载全部 split 字节，但脚本不索引或评价 test。

## 六次候选

| 模型 | 种子 | 运行轮数 | 检查点轮次 | 最优 valid J |
|---|---:|---:|---:|---:|
| B0 | 2029 | 10 | 2 | 0.246163147 |
| B0 | 2030 | 10 | 2 | 0.259831341 |
| B0 | 2031 | 12 | 4 | 0.252691635 |
| B1 | 2029 | 10 | 2 | 0.264148854 |
| B1 | 2030 | 10 | 2 | 0.251689448 |
| B1 | 2031 | 12 | 4 | 0.258133775 |

`training_progress.json` 是第六条完成后写入的**中间快照**，其 `status=IN_PROGRESS` 属于脚本写入顺序；它已列出六条候选，最终状态以随后写入的 `training_summary.json` 的 `B0_B1_VALID_SELECTION_COMPLETE` 为准。

## 所选检查点 valid 复算

| 指标 | CPU 复算值 | 与服务器记录的绝对差 |
|---|---:|---:|
| Accuracy | 0.644230769 | 0 |
| Macro-F1 | 0.607073787 | 0 |
| MAE | 0.596400484 | 约 1.23e-9 |
| Selection J | 0.246163147 | 约 1.03e-10 |
| Pearson | 0.629074012 | 另见 `valid_replay.json` |

复算达到数值一致。`training_summary.json` 同时记录 `test_used_for_selection=false` 和 `attachment4_used_for_selection=false`；训练入口源码只取 train/valid 分组，六个候选的模型选择没有读取 test 指标。

## 冻结对象与边界

`MODEL_FREEZE_MANIFEST.json` 锁定所选检查点、服务器训练 scaler、数据原件、资格门、解释合同和关键源码的 SHA-256。所选检查点 SHA-256 为 `723a9ddef831f35c25d95b325a51c194a8e7326460812ab5435c0b24fc8ad6ce`；服务器 `train_scaler.pt` 为 `57d93f8d17fce56b928fadb1039bbe382a456386e39477980d44b43a2afd1424`。本次公开审核包包含完整指标历史、审核脚本和哈希，不包含六个二进制检查点或 `train_scaler.pt`；原件保留于 DR-X 和本机下载目录。

公开审核材料足以核对训练结果和文件身份，但不能单靠哈希重现模型输出；本机复算使用了下载的所选检查点和从 train 原件重新拟合的 scaler，并确认服务器保存的 scaler 数组与重新拟合结果完全相同。归档哈希是本机下载后计算，未获得服务器独立生成的归档哈希；ZIP CRC、回执、检查点内容与 valid 复算提供了交叉核验。

下一步应在**单独阶段**核验冻结模型的 Shapley、conditional IG 和 XAI validation，再决定是否放行最终 test 评价与附件4正式推理。解释边界仍为附件4指定20条文本可回溯词片段；音频、视觉仅允许特征空间索引归因。
