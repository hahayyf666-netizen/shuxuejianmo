# Q3 4.6 独立红队审核报告

**结论：PASS_WITH_LIMITATIONS。** 本轮独立、只读审核没有未关闭的 P0、P1 或 P2 问题。审核对象是已冻结的 B0 模型、附件4正式20条结果、727条 test 评价、C2/T2/T3 映射记录，以及补验后的4.5回执。没有重新训练或推理，也没有用 test 调整模型、阈值或解释规则。

## 核查结果

| 核查项 | 独立结果 |
| --- | --- |
| test 样本 | 727/727 个唯一编号；从逐样本预测和标签重新计算指标、混淆矩阵和三类 F1，均与摘要一致 |
| test 指标 | Accuracy 0.6588720770；Macro-F1 0.5824303167；MAE 0.6475765522；RMSE 0.8727662373；Pearson 0.6603898510 |
| 模型身份 | valid 阶段选择 B0 seed2029；附件4与 test 使用的 checkpoint 和 train scaler SHA 一致；未见 test 参与选模 |
| 附件4 | 20/20 条有正式预测及解释，160项数值检查无失败；29个输出文件 SHA 与两个 manifest 一致 |
| 来源行 | C2 与 T2 的1128条 sample/modality/index 键及唯一来源索引逐行一致；音频564条唯一，视觉522条唯一、1条非唯一、41条 C2 断链 |
| 媒体起点 | T3 的15个候选中9个媒体起点完整通过；不能把这9个结果推广到其余样本或直接称为正式音视频原素材解释 |
| 解释边界 | 文本 `verified_text`；音频、视觉 `index_only`，未输出未经验证的秒数、PTS或关键帧 |

## 审核发现与处理

审核发现4.5脚本原先只检查 C2/T2 行数，现已加上1128条逐键、逐唯一来源索引检查并通过。正式18号结果的 `known_input_anomaly=T2 vision row nonunique` 容易造成误读。单独勘误 `metadata_erratum.json` 记录了原样本 JSON 与预测 CSV 的 SHA，并说明：二义发生在一条未对齐原生视觉行；18号模型使用的48条 aligned 视觉内容行均有唯一来源。冻结合同、预测、Shapley、conditional IG、正式结果文件和 manifest 均未改动。18号视觉仍为 `index_only`，因为其附件4本地媒体帧映射没有通过正式验证。

## 边界与阶段决定

本轮复算的是已经输出的727条 test 预测与标签，没有重新加载约1 GB 的 PKL 做第二次模型推理。现有代码、训练摘要和产物时间顺序未显示 test 选模；仓库外的历史行为无法据此作绝对证明。附件4没有标签，因此不报告附件4预测准确率或“解释准确率”。

**4.6 Gate：PASS_WITH_LIMITATIONS。** 允许另行决定是否进入4.7封存；本轮没有执行4.7。

关联文件：`../final_4_5_review/q3_4_5_final_review.json`、`../final_4_5_review/metadata_erratum.json`、`../final_4_5_review/Q3_4_5_FINAL_REVIEW_REPORT.md`。

