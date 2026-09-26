# Q3 4.5 最终内部复核

**状态：PASS_WITH_LIMITATIONS**。新增 test 评价后，独立重算727条逐样本结果：Accuracy=0.658872，Macro-F1=0.582430，MAE=0.647577，RMSE=0.872766，Pearson=0.660390；全部与冻结测试摘要一致。

## 解释证据边界

`c2_t2_t3_mapping_status_20.csv` 汇总20条样本、564个内容位置：音频来源行564/564唯一，视觉来源行522/564唯一，1行原生来源二义，41行C2链不可用。T3有9条完整媒体起点通过。上述证据没有把官方模型输入行完整验证到附件4本地音频时段或视觉帧，因此正式状态继续为文本 `verified_text`，音频和视觉 `index_only`；逐样本明确 `raw_evidence_unavailable=audio|vision（仅指音视频）`。

06的视觉来源行二义；13、16的视觉C2链断开；18的原生未对齐视觉行二义但不在模型aligned内容行。异常样本均保留在正式20条输出。

## 测试与产物核验

本次只读核对了冻结checkpoint、train scaler、输入数据SHA、727条唯一test样本、混淆矩阵、Attachment4和test输出清单。test未参与架构或阈值选择；附件4预测与解释工件未改变。共15项核查，失败0项。完整逐项证据见 `q3_4_5_final_review.json`。

## 阶段结论

4.5最终内部复核达到 `PASS_WITH_LIMITATIONS`。它允许把当前冻结产物提交4.6独立审核，但不构成独立审核通过或4.7最终封存。音频秒级证据、视觉关键帧、因果解释和“解释准确率”仍不属于当前正式交付范围。

## 冻结输出注释勘误

正式结果里18号的 `known_input_anomaly=T2 vision row nonunique` 表述过宽。准确含义是：一条未对齐原生视觉来源行有二义，18号模型使用的48条 aligned 视觉内容行均有唯一来源行。该勘误见 `metadata_erratum.json`，其中记录原样本文件及预测 CSV 的 SHA；正式结果、合同和 manifest 保持原样。18号视觉正式解释仍为 `index_only`，因为附件4本地媒体时间或帧映射未通过合同验证。
