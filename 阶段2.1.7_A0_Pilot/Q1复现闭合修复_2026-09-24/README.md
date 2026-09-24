# Q1 v1 第一版功能闭合交付

日期：2026-09-24  
结论：**作者侧闭合检查通过；独立P2只读验收PASS。保持2.1.7 REVIEW GATE，未进入2.2，也未标记Q1最终冻结。**

## 最终候选包

- [Q1 v1 候选附件包](package/q1_v1_candidate_q1_closed_2026-09-24.zip)
- SHA-256：`ba0e0c52057cb810441909cdfd0c23330e8bb4ad0e0d7b3e27ed2c9bd3e70ca0`
- 字节数：27,934,733；Q1 ZIP 单独距 50,000,000 bytes 还有 22,065,267 bytes。该数不是最终全题附件预算。
- SHA、大小与成员清单 sidecar：`package/q1_closed_archive.sha256`、`package/q1_closed_size_report.json`、`package/q1_closed_package_manifest.json`。

## 审计结论

- 全部100条样本、100个NPZ、manifest/results一一对应；最终归档全量结构及题目需求复审100/100通过，ZIP CRC及成员SHA/集合均通过。
- 分支数量：词级映射5条，官方文本与音频不对应但A/V保留5条，数字静音2条，clip-only文本与原生A/V序列保留88条。88条没有伪造词时间；ASR是机器筛查，不作为人工转写真值或逐条听辨待办。
- 最终SHA对应的全量回执与逐样本表在`reports/final_package_recheck_2026-09-24/`。独立P2报告在候选包的 `reports/P2独立只读验收_2026-09-24.md`。
- 9条冻结分支流水线和5条已有词级trace的命令、日志与逐样本结果在 `reports/final_archive_recheck_2026-09-24/`。这些复验运行所用候选ZIP SHA为`d3c82f92a214375736fe6c4eb6ba0e6c99b508508a054b9f808c16ceef224146`；与本最终包对比时，生产代码、配置、正式特征、manifest和输入指纹成员0变化。差异报告在包外最终复验目录。

## 文件索引

- `Q1第一版闭合验收报告.md`：原题功能闭合结论、模式计数与限制。
- `reports/P2独立只读验收_2026-09-24.md`：独立验收证据与结果边界。
- `reports/portable_env_clean/`：便携CPU环境安装、固定依赖与资产哈希凭据。
- `reports/final_archive_recheck_2026-09-24/`：9条和5条重放材料。
- `reports/final_package_recheck_2026-09-24/`：最终候选ZIP的100条验收回执和逐样本检查表。
- `Q1闭合推进步骤_执行记录.md`：从审核门到第一版闭合的流程记录。

本交付仅表示Q1 v1功能闭合，不包含Q2/Q3，不表示进入2.2或Q1最终冻结。
