# Stage C-2：本批 PKL 特征来源调查

只读核验附件4对齐版、未对齐版及同名原 MP4 的实际数值与时间元数据。赛题 DOCX 是本批附件的字段合同；MMSA 通用说明没有被当作附件4的特征生成来源证明。报告见 `c2_provenance_report.md`。

## 复现

在仓库根目录，使用带 NumPy、Transformers 的 Python 3.12 环境运行：

```powershell
python '阶段4.2_Q3_v1_preflight/stage_c2_provenance/audit_pairwise_sources.py' `
  --aligned '<附件4>/对齐版本' --unaligned '<附件4>/未对齐版本' `
  --problem-docx '复杂场景下多模态情感识别的数学建模与算法设计.docx' `
  --out '阶段4.2_Q3_v1_preflight/stage_c2_provenance/results'

python '阶段4.2_Q3_v1_preflight/stage_c2_provenance/audit_media_pts.py' `
  --ffprobe '<ffprobe.exe>' --video-directory '<附件4>/对齐版本/videos' `
  --pair-inventory '阶段4.2_Q3_v1_preflight/stage_c2_provenance/results/pair_inventory_20.csv' `
  --out '阶段4.2_Q3_v1_preflight/stage_c2_provenance/results'

python '阶段4.2_Q3_v1_preflight/stage_c2_provenance/verify_c2_artifacts.py' `
  --aligned '<附件4>/对齐版本' --unaligned '<附件4>/未对齐版本' `
  --results '阶段4.2_Q3_v1_preflight/stage_c2_provenance/results'
```

第一步逐样本计算 PKL/MP4 SHA、跨版本文本与媒体字节一致性，并把对齐版每个内容行与未对齐版有效长度内的数值行作**完全相等**匹配。第二步只解码原视频的音频/视频 PTS，并记录格式、帧数及结果哈希。时间戳只归属于原媒体；脚本没有把未对齐特征索引换算为秒或帧，也没有加载 Q3 预测模型。

第三步重新打开原 PKL，逐行复核公开的 1128 条数值匹配记录，检查 PTS 汇总和未通过门的状态。主要文件：`results/pair_inventory_20.csv`、`results/aligned_to_unaligned_row_matches_20.csv`、`results/vision_unmatched_rows.csv`、`results/media_pts_20.csv`、两个汇总 JSON、`results/c2_verification.json`、`c2_gate.json`。
