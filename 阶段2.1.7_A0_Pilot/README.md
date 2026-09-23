# 阶段 2.1.7：冻结三条 A0 pilot

审核状态：REVIEW GATE。本目录归档三条冻结 pilot 的特征产物、对齐结果、试听材料和机器审计。当前不判定 A0 pilot 整体通过，不进入 2.2。

## 样本与结果

| sample_key | 角色 | 输出词数 | 结果 |
| --- | --- | ---: | --- |
| -s9qJ7ATP7w$_$0 | ordinary | 24 | SAMPLE FAIL：第 4 个输出词 they 的区间为 [0.780, 0.780)；保留该词，不伪造边界。 |
| -mJ2ud6oKI8$_$6 | edit-list | 5 | COMPLETED：5/5 词通过对齐及三模态特征检查。 |
| -s9qJ7ATP7w$_$6 | edit-list | 5 | COMPLETED：5/5 词通过对齐及三模态特征检查。 |

样本选择清单见 pilot_samples.json。整体执行说明见 执行报告.md。

## 文件索引

- manifest.csv、run_summary.json、artifact_audit.json：结果索引、运行摘要和结构审计。
- environment_and_assets.json、input_timeline_preflight.json：环境/模型资产与输入时间轴记录。
- alignment/：逐词对齐表及完整映射 trace。
- npz/：逐词特征与有效性掩码；结构审计确认文件可在 allow_pickle=False 下读取，未包含情绪目标标签。
- cache/：流水线缓存。
- review_audio/：供人工复核的三条 16 kHz 单声道音频。
- logs/：逐样本运行记录；logs/attempt_1/ 保存可恢复故障及重跑前的证据。
- recovery_log.json：恢复分类及修复说明。

## 复核边界

普通样本一个词的 forced-alignment 区间长度为零，已标记 SAMPLE FAIL。三段试听音频仍需人工对照官方文本和词级对齐表核验；这一步尚未完成。结构审计通过不代表时间边界或内容正确性已经人工确认。

特征维度为文本 768、音频 50、视觉 104。执行采用共同 presentation timeline、解码帧 PTS 和半开区间 [start, end)；具体定义及异常见 执行报告.md 与审计文件。

上传副本中的本机绝对路径已替换为占位符或相对路径；数值产物与二进制特征/音频文件保持原样。