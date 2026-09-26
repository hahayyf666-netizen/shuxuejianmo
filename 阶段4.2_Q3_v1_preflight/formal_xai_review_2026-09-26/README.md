# Q3 valid120 正式解释验证审核材料

- `Q3_VALID120_XAI_REVIEW_REPORT.md`：审核结论、统计范围和限制。
- `REVIEW_GATE.json`：机器结果门与人工审核决定分别记录。
- `primary_audit.json`：从逐样本结果重新核算的结构和统计回执。
- `audit_valid120_result.py`：只读复核脚本，不修改模型或服务器结果。

原始服务器结果 ZIP SHA-256 为 `33ef931cfacfee17ce36746b975e4d0440a7aec5371ddb616ad32193876b6645`。原 ZIP 含运行环境绝对路径和逐样本结果，保存在参赛队本地工作区，未放入公开仓库。公开的报告和回执不包含服务器连接信息。若需逐行独立复核，应在受控工作区取得与该 SHA 一致的原 ZIP、解包，使用冻结运行包中的 valid120 清单和 `validation_contract.json` 调用审核脚本。

本轮独立代理审查因工作区额度不足未完成。`REVIEW_GATE.json` 中的 `PASS_WITH_LIMITATIONS` 是本轮内部审核决定，不等于 4.6 独立红队通过或 Q3 冻结。
