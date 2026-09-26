# Q3 4.5 内部验收报告

**验收门**：`Q3_4_5_INTERNAL_ACCEPTANCE`  
**结果**：`PASS_WITH_LIMITATIONS`  
**检查数**：36；失败检查：0  
**停止点**：本报告完成后停止，不进入 4.6 独立红队复核，也不进入 4.7 最终冻结。

## 1. 完成了什么

1. 完成 T3 有界媒体起点核验：15 条冻结候选中，9 条音频+视频完整通过，11 条音频门通过，9 条视频门通过；T3 没有读取标签、测试集、模型预测或 XAI 输出。
2. 完成 Attachment4 01–20 的正式预测与 feature-space attribution 输出：20 条预测、40 条 exact 3-player Shapley、120 条 conditional IG 数值检查，全部通过。
3. 生成一行一条的 `predictions_explanations.csv`、逐样本 JSON、数值检查、输出 SHA manifest，并完成只读一致性审计。

## 2. 得到的结论

- T3 媒体导航证据在冻结候选范围内达到有限目标，但它不等于官方特征提取 provenance，也不改变正式解释映射合同。
- 正式解释结果的映射边界保持为：`text=verified_text`、`audio=index_only`、`vision=index_only`。
- Shapley/conditional IG 的数值闭合、概率单纯形、样本覆盖、输入 SHA、输出 manifest 和禁止测试集/标签读取检查均通过。
- `06、13、16、18` 的已知输入异常被保留；没有因为异常删除样本或伪造音视频证据。

## 3. 仍未确认的内容

- T3 未建立每一条官方 audio/vision feature row 到附件4本地秒数或视频帧的正式合同映射，因此不能报告音频秒级证据、视觉关键帧或官方特征采样窗口。
- T3 中 03、05、12、20 被来源媒体元数据条件阻断，13、18 只有音频门通过而视频门未通过；这不被解释为来源不存在。
- 这些限制属于证据范围限制，不是本次内部验收的数值失败。

## 4. 是否满足进入下一阶段的条件

`4.5` 内部验收条件满足，状态为 `PASS_WITH_LIMITATIONS`。本次按指令在 4.5 停止；是否进入 4.6 由后续单独指令决定。

## 5. 主要复现入口

- T3：`t3_media_origin_2026-09-26/run_t3_media_origin.py`
- T3 规则与结果：`t3_media_origin_2026-09-26/results_v3/t3_frozen_contract.json`、`t3_aggregate_gate.json`、`t3_media_origin_records.json`
- Attachment4 正式入口：`formal_attachment4_2026-09-26/run_attachment4_formal.py`
- 本次验收入口：`formal_attachment4_2026-09-26/run_internal_acceptance.py`
- 结果目录：`formal_attachment4_2026-09-26/results/`

完整逐项检查结果保存在同目录 `internal_acceptance.json`。私有来源媒体及下载命令不纳入公共报告。
