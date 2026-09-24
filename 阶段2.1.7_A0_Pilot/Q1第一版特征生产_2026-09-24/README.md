# Q1 v1 第一版特征生产与步骤09闭合审核

本目录归档 Q1 第一版全量100条特征生产、审计、复现、候选包和步骤09需求追踪材料。

## 结果摘要

- 100/100 样本保留并生成特征；100 个 NPZ 均通过可读性和包副本验证。
- alignment_mode：TRI_MODAL_WORD_VALID 5，TRI_MODAL_WITH_AUDIO_CONTENT_INVALID 2，AV_VALID_TEXT_UNALIGNED 5，UNCERTAIN_REVIEW 88。
- 88 条 UNCERTAIN_REVIEW 表示未作文本—语音对应断言，不是待用户逐条复核任务，也不代表已确认文本匹配。
- 静音样本保留真实声学窗口；无脸样本保留视频帧与共享时间轴。
- **2026-09-24 原题复查：NEEDS_REVISION。** 100条覆盖、结构及当前规则下的时间组织通过；仍有2项P1复现交付缺口：依赖锁绑定原电脑路径，上游词对齐生成代码未完整交付。旧步骤09整体PASS结论已修正。保持2.1.7 REVIEW GATE，不进入2.2。
- Q1 候选包：23,623,432 bytes；SHA-256：d1b3bb587bd96b4094b555eeab1add4eed048c316ee84fbb6d8a3456fa39d7a2。ZIP 成员CRC、成员SHA、100/100副本校验和特征读取均通过。包清单与容量报告位于 package/，详细报告位于 reports/。
- 按50,000,000-byte全题总上限粗算，当前Q1包后剩余26,376,568 bytes；Q2/Q3及其他材料合并后的最终总量尚未验收。

## 目录

- reports/：步骤06/08/09审计、方法与结果、复现和验证报告
- contracts/：特征合同、字段规范、旧新规则差异
- manifest/：100条输出manifest和输入清单
- code/：生产、读取、审计、复现及样例生成代码
- environment/、metadata/：依赖、配置、资源和来源证据
- examples/：典型跨模态对应核验
- package/q1_v1_candidate.zip：本次被审候选包；存在新报告列出的复现交付缺口，尚未重建

最新结论请先看 [Q1原题功能对齐检查](reports/requirement_alignment_check_2026-09-24/Q1原题功能对齐检查.md)，附100条检查表、日志与证据ZIP。步骤09旧报告已添加结论修正，旧ZIP保留用于追溯。原始MP4、label-100.xlsx及大模型权重不放入仓库；取得方式和哈希见 metadata/。

