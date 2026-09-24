# P2 最终归档身份复核

日期：2026-09-24  
结论：**PASS**。这次只复核最终归档身份和与已审候选包的成员差异；未改文件，也未重跑特征。

- 最终ZIP SHA-256：`ba0e0c52057cb810441909cdfd0c23330e8bb4ad0e0d7b3e27ed2c9bd3e70ca0`，与指定最终包一致。
- 最终归档CRC通过。包外回执记录1824个成员、100/100全量复审通过、0失败、manifest/results一致、成员集合与所有成员SHA匹配。
- 相对P2已审阅的前一候选包SHA `458ab0036d6160c6c0f6c17bb96f27ea9f144a7ed676929ba2e4390b5888d929`：新增1个P2验收回执、更新2个说明文档、删除0个文件；其余1823个成员逐成员SHA完全相同。
- 生产代码、配置、输入指纹、正式特征、manifest与results均未变化；`replay_relevant_members_unchanged=true`。九条分支和五条词级trace复现凭据可适用于最终候选包。
- 与复现包`d3c82f92a214375736fe6c4eb6ba0e6c99b508508a054b9f808c16ceef224146`相比，包外差异报告记录新增28项、更新2项、删除0项；新增/改动均为闭合说明或复验凭据，生产相关成员未变。

详细机器回执与逐样本表见同目录的`final_package_receipt.json`、`final_archive_member_diff.json`及`final_requirement_audit/`。
