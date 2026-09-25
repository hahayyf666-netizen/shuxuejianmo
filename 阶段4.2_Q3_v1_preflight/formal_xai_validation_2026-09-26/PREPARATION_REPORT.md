# Q3 正式解释验证运行准备报告（2026-09-26）

## 目标与范围

本目录实现冻结 B0_seed2029 的解释验证入口，严格使用附件2 train/valid、预冻结的 valid120 样本、已审计 checkpoint 和 scaler。该阶段只准备服务器运行包并验证首条样本；完整120条结果须在 DR-X 运行后单独审核。历史 C-1 至 C-4、原 Q3 模型、训练、Shapley 和 conditional IG 源码均未修改。

附件2 valid 三模态的 raw evidence mapping 状态均为 `index_only`。这里的解释范围是 feature-space attribution：模态级三玩家 Shapley、官方序列索引和模态特征位置上的 conditional IG。不得将其解释为原词、音频秒数、视频帧、因果作用或“解释准确率”。

## 冻结身份

- 训练审计源提交：`dfe531592d3eaefaa8c8b664f45d1d240d2dc008`。
- 选定模型：B0_seed2029；checkpoint SHA-256 `723a9ddef831f35c25d95b325a51c194a8e7326460812ab5435c0b24fc8ad6ce`。
- scaler SHA-256 `57d93f8d17fce56b928fadb1039bbe382a456386e39477980d44b43a2afd1424`。
- valid120 ID 清单 SHA-256 `aeaef63ae3906153a88bf58e9e3653edad3d2d32c95f11b9adf30c252d81991e`。
- 解释参数和随机流见 `validation_contract.json`。运行时再次核对数据、模型、scaler、训练摘要、源文件和样本清单哈希。

## 方法与边界

- 三模态全部8个联盟精确计算 Shapley；分类目标固定为完整输入下 B0_seed2029 预测类的概率，回归目标为原输出。
- 每一模态 conditional IG 只将该模态从参考移动到实值，其他模态保持实值；64/128/256 点 Gauss–Legendre，完备性阈值 `1e-3 + 0.01 × |输出差|`。
- train 均值和 train 中位数参考均在冻结标准化空间计算；中位数只用 train 有效内容位置。B0_seed2030/2031 与全参数随机化模型仅用于排序稳定性诊断，分类比较始终固定原预测类。
- 位置排名用“各通道有符号 IG 之和的绝对值”，分别替代10%、20%、30%有效位置，与50组等数量的确定性随机位置比较；正向分类支持额外记录有符号概率下降。
- video_id 分组配对 bootstrap 2000 次报告95%区间。样本不足时明确 `insufficient_groups`，数值PASS不自动转成有效性PASS。弱归因与未定位交互不强制产生Top-k。
- 只准 `valid` 解释验证，不使用 `test` 做模型选择，不运行附件4正式推理；所有完整结果停在 `REVIEW_GATE`。

## 准备检查

- 本地 Python3.12、numpy2.5.0、torch2.8.0+cpu，8项针对性单元测试通过。
- 冻结列表首样本 `PyQrAYl1bFs$_$4` 完整烟测成功：34项数值检查通过、0失败；epsilon 重复检查通过；状态 `SMOKE_COMPLETE / REVIEW_GATE`。
- 独立 P1 审查 PASS：核对固定类别目标、真实模型前向、全部扰动位置与响应、分组 bootstrap、未定位交互和输入身份；P0/P1 无。独立审查的源码 SHA 和证据位于本地 `outputs/Q3_xai_validation_preparation_2026-09-26/independent_p1/`。
- 独立 P2 解包复现 PASS：包内20个文件和清单校验通过，解包后8项测试与首条 valid 烟测通过；逐样本输出与P1一致，新增汇总独立重算一致。未发现 P0/P1/P2 问题。P2原始证据留在本地 `outputs/Q3_xai_validation_preparation_2026-09-26/independent_p2/`。
- 冻结 ZIP 来源提交 `a6b6bb552613cd4c59ab23e99f15b5b8a920a082`，SHA-256 `0d085aa7cc87f4c8210ba3972e3d45034313231015dfcac19e3df067e45c797b`，含包内清单在内共21个文件；正式120条尚未运行。

## 交接与完成条件

打包脚本只接受与提交源码逐字节一致的输入，ZIP 内含文件 SHA 清单，外部清单记录 ZIP SHA。服务器说明见 `README_SERVER.md`。正式运行必须产出120条样本、数值检查、扰动和敏感性汇总、逐样本证据、进度与完整文件清单；再人工审核效果和限制，方可决定后续 test/附件4 范围。当前状态为**服务器运行准备**，不是120条解释有效性结论。
