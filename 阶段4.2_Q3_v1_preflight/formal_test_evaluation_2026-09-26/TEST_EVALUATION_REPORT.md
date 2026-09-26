# 冻结模型最终 Test 评估报告

**状态**：`FINAL_TEST_EVALUATION_COMPLETE`  
**模型**：`B0_seed2029`  
**评估集**：附件2 `test`，共 727 条  
**模型选择/调参**：未使用 test；本次只做一次冻结模型评估。  
**附件4结果**：未修改。  

## 指标

| 指标 | 值 |
|---|---:|
| Accuracy | 0.658872 |
| Macro-F1 | 0.582430 |
| Negative F1 | 0.653563 |
| Neutral F1 | 0.329317 |
| Positive F1 | 0.764411 |
| MAE | 0.647577 |
| RMSE | 0.872766 |
| Pearson | 0.6603898509560031 |
| 选择指标 J（仅记录，不重新选择） | 0.262750 |

混淆矩阵行列顺序均为 `Negative / Neutral / Positive`，详见 `test_confusion_matrix.json`。逐样本真值与预测详见本地 `test_predictions.csv`，不作为附件4解释结果发布。

## 约束核验

- 使用已冻结的 `B0_seed2029`，没有重新训练；
- scaler 的来源为 train；
- test 没有用于模型、架构、阈值或解释规则选择；
- 没有修改附件4预测、Shapley 或 Conditional IG 文件；
- 本报告是性能评估，不改变音频/视觉 `index_only` 的证据范围。

