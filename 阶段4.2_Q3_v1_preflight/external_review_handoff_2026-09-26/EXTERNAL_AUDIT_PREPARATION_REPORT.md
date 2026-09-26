# Q3 外部4.6审核准备报告

日期：2026-09-26。当前 Gate 继续为 `PENDING_EXTERNAL_REVIEW`。本报告记录工件核验与可访问性，不替代外部审核结论。

## 已核验的历史工件

| 工件 | 本地 SHA-256 | 预期 SHA-256 | 结果 |
| --- | --- | --- | --- |
| test 逐样本 CSV | `454e62de2c3704823f65bb13bb92ae04bff9d8a205e28e0428ee4d7a0f11b3b5` | 同左 | MATCH |
| test output manifest | `40dd2b2ec6ded10d04f5cb8f616f8b5c5ce8af599816388007d3da1074223a61` | 同左 | MATCH |
| Attachment4 predictions CSV | `e219309f0c8460ef34e2ae597e421df00ab9579220b8fa5066422a78acea414c` | 同左 | MATCH |
| Attachment4 output manifest | `ba7902c2193ed53a677b043a8ec56d61a7c087ce76c9e1470a3c6a290d6bdc1a` | 同左 | MATCH |
| Attachment4 explanation contract | `f9a7e448771929e242e6c0b96d61bca053537fbf9c1f15b634b1316e5c47c794` | 同左 | MATCH |
| valid120 正式结果 ZIP | `33ef931cfacfee17ce36746b975e4d0440a7aec5371ddb616ad32193876b6645` | 同左 | MATCH；120个样本文件、129项内部 manifest 全部匹配 |

另外，Attachment4 本地正式结果目录的25个 manifest SHA 均通过逐文件校验。正式数据记录和文件未被覆盖。

## 只读重算结果

- test 审计脚本从727行现存 CSV 独立重算并通过：sample ID唯一；概率和、强度有限值通过；Accuracy 0.6588720770、Macro-F1 0.5824303167、MAE 0.6475765522、RMSE 0.8727662373、Pearson 0.6603898510、逐类 F1 和混淆矩阵与摘要一致。
- Attachment4 审计脚本核验20条、概率单纯形20/20、强度范围20/20、Shapley加和40/40、conditional IG完备性120/120、Top位置没有越过有效内容范围、异常样本06/13/16/18仍保留、文本 verified_text，音频与视觉 index_only。
- 主要模态原始证据覆盖：分类18/20、回归18/20满足；未满足的主模态均为视觉。具体逐条表和结果 JSON 在本地审核包，未放入公共分支。正式 schema 没有定义回归 `primary_supporting_modality`，不推造此统计。
- T3 15个候选：audio PASS 11、video PASS 9、完整 media identity PASS 9；这些结果不升级正式 audio/vision 映射。

## CSV 生成链结论

`export_attachment4_csv_from_frozen_json.py` 只读取冻结 `attachment4_predictions_explanations.json`，不加载模型、不推理、不覆盖现存文件；在临时目录生成的 CSV 与正式历史 CSV 字节完全一致，SHA-256 为 `e219309f0c8460ef34e2ae597e421df00ab9579220b8fa5066422a78acea414c`。因此可以确定该 CSV 可由冻结 JSON 确定性导出。**没有同期日志或冻结源码提交证明当时实际调用了哪份 exporter**；历史执行路径标记为 `UNVERIFIED_HISTORICAL_SOURCE`，不猜测。候选正式 runner 也没有被历史 source bundle 绑定。

## 外部审核材料可访问性

公开分支已包含汇总报告、复现代码、合同摘要、C2/T2/T3 证据索引、SHA 清单及本 handoff 指引。GitHub 写入审核拒绝了本次 test CSV 上传，明确指出提交正文可能被截断，无法证明其与冻结原文件字节一致。按拒绝提示，没有再用分卷、编码或其他方式间接发布该文件。valid120 ZIP 和20条 Attachment4逐样本结果也未公开上传。因此审核者仅凭公开 GitHub **不能独立重算 test 指标或逐条复核 Attachment4**，外部 Gate 仍 `BLOCKED_BY_MISSING_ARTIFACT` / `PENDING_EXTERNAL_REVIEW`。

完整原始材料已打成本地外部审核包，由用户安全地提供给指定审核者。约1 GB 原始 aligned PKL 和原始 MP4 未包含；本次审核不要求重推理，若审核者认为必须重新验证来源输入，应另行提供这些输入及核 SHA。

## 操作边界

- 重新训练：NO。
- 重新运行模型推理：NO。
- 使用 test 调参或选模：NO。
- 修改 B0/scaler：NO。
- 修改正式预测、Shapley、IG数值：NO。
- 修改正式解释范围：NO。
- 外部4.6审核：尚未完成；等待用户指定的另一位 AI。
- 4.7封存：未执行。

