# Q3 外部独立审核阅读指引

## 审核身份与当前状态

请以**从未参与本项目的外部审核者**身份，从原题和附件说明独立判断第三问要求，再检查本分支的实现与结果。不要把报告中的 `PASS` 当成证据。此前由同一 Codex 任务组织的子代理复核仅属内部隔离交叉检查，不替代外部审核。当前 finalization candidate 已补齐公开审阅所需的冻结 test CSV、Attachment4逐样本结果、valid120原始ZIP展开文件、checkpoint/scaler及派生报告；candidate 状态为 `READY_FOR_EXTERNAL_REVIEW`。外部4.6尚未完成，不能写为 PASS；未经独立外部结论，不进入4.7。

固定阅读分支：[`q3-v1-preflight`](https://github.com/hahayyf666-netizen/shuxuejianmo/tree/q3-v1-preflight)。原题、数据说明从仓库根目录和 `main` 中读取；Q3 实现以本分支的 `阶段4.2_Q3_v1_preflight/` 为准。该目录顶层 `README.md` 记录的是较早阶段状态，**不能作为最终进度依据**。

## 当前 finalization candidate（2026-09-26）

请先读 `q3_finalization_candidate_2026-09-26/README.md` 和 `review/Q3_4_6_R1_CANDIDATE.json`，再核查候选代码、结果和图。候选结论为 `READY_FOR_EXTERNAL_REVIEW`，不代表外部 Gate 已通过。历史 `q3_4_6_gate.json` 保持原样；它记录的 `BLOCKED_BY_MISSING_ARTIFACT` 是更早的状态，不由本次候选文档覆盖。

当前 public branch 已包含：
- 727条 `test_predictions.csv` 及其 manifest/摘要；
- Attachment4 20条冻结预测与归因 JSON/CSV、valid120展开结果；
- 冻结 checkpoint 与 train scaler；
- T4-aware最终派生 CSV、valid728误差分析、test冲突分析、7张论文图、valid/test/Attachment4冻结结果复现报告。

当前候选重新计算的 Attachment4 原素材证据覆盖为：分类主要参考模态 20/20、分类主导影响模态 20/20、回归主导影响模态 19/20。test冲突统计为22/727，其中低强度21、低 margin 20；仅作描述，不调整预测。valid728核心指标在1e-6容差内复现 checkpoint 记录。T4仍为 `T4_COMPLETE_WITH_LIMITATIONS`，样本05视觉仍为 `feature_position_only`。完整原始 `aligned_50.pkl`（约1 GB）不在公开 GitHub，故无法单靠公共分支重跑完整模型推理。

**独立审核任务：** 从原题建立要求—实现—产物证据矩阵；独立复算公开 test/Attachment4 记录，检查训练分离、XAI恒等式、T4边界、异常保留与复现限制。必须区分独立重算、核对摘要、沿用既有报告。审核者应自行给出4.6结论；本任务不要求、也未预写最终 Gate。


## 已完成工作的索引

1. **数据合同与模型。** 预计算的 `aligned_50` 特征进入 B0/B1；文本预测输入为 `text`，`text_bert` 用于映射检查。先读 [`frozen_config.json`](../frozen_config.json)、[`q3v1/data.py`](../q3v1/data.py)、[`q3v1/model.py`](../q3v1/model.py)、[`q3v1/train_eval.py`](../q3v1/train_eval.py)，再读 [`run_formal_training.py`](../run_formal_training.py)。
2. **服务器和训练。** DR-X 的 CUDA preflight 通过；B0/B1 各3个种子训练；用 valid 选择 B0 seed2029，报告 test 未参与选择。证据见 [`服务器资格门`](../server_preflight_drx_cuda128_2026-09-26/SERVER_PREFLIGHT_CUDA128_AUDIT.md)、[`训练审计`](../formal_training_audit_drx_cuda128_2026-09-26/FORMAL_TRAINING_AUDIT_REPORT.md)、[`训练摘要`](../formal_training_audit_drx_cuda128_2026-09-26/training_summary.json)、[`模型冻结清单`](../formal_training_audit_drx_cuda128_2026-09-26/MODEL_FREEZE_MANIFEST.json)。
3. **解释方法与 valid 验证。** 模态级 exact 3-player Shapley、特征空间 conditional IG；valid120 的正式数值验证与预定扰动对照见 [`解释实现`](../q3v1/explain.py)、[`valid120 审核`](../formal_xai_review_2026-09-26/Q3_VALID120_XAI_REVIEW_REPORT.md)、[`解释范围合同`](../formal_xai_validation_2026-09-26/assets/explanation_scope_contract.json)。不要把特征空间重要性写成原素材因果证据。
4. **原素材映射边界。** C2/T2 建立官方 aligned 行到 unaligned/来源特征行的链；T3 在部分样本上验证媒体起点。请查 [`C2`](../stage_c2_provenance/c2_gate.json)、[`T2`](../t2_row_boundary_and_source_inventory_2026-09-26/results/t2_aggregate_gate.json)、[`T3`](../t3_media_origin_2026-09-26/results_v3/t3_aggregate_gate.json) 和 [`4.5 最终复核`](../formal_attachment4_2026-09-26/final_4_5_review/Q3_4_5_FINAL_REVIEW_REPORT.md)。正式解释边界仍是文本 `verified_text`（附件4范围），音频和视觉 `index_only`；T3 局部成功不能自动升级正式解释。
5. **附件4和 test。** 附件4有20条正式预测、强度、Shapley、局部IG及映射状态；报告见 [`4.5 首轮验收`](../formal_attachment4_2026-09-26/Q3_4_5_INTERNAL_ACCEPTANCE_REPORT.md) 和 [`最终4.5回执`](../formal_attachment4_2026-09-26/final_4_5_review/q3_4_5_final_review.json)。冻结 B0 后对附件2 test 作一次评价，727条；摘要见 [`test 报告`](../formal_test_evaluation_2026-09-26/TEST_EVALUATION_REPORT.md)、[`test 摘要`](../formal_test_evaluation_2026-09-26/test_evaluation_summary.json)。附件4无标签，不能报告附件4准确率。
6. **历史注释勘误。** 18号样本的未对齐原生视觉行有一处来源二义；其48条模型 aligned 内容行均唯一。见 [`metadata_erratum.json`](../formal_attachment4_2026-09-26/final_4_5_review/metadata_erratum.json)。冻结正式结果及 SHA 未被改写。

## 你需要独立回答的问题

1. 从原题逐项建立“要求 → 代码 → 产物 → 可验证证据”矩阵。尤其检查分类、强度、模态贡献、局部证据、验证以及附件4交付是否都满足；不要把实现者的分阶段命名当成题意。
2. 检查 train/valid/test 分离：scaler 是否仅从 train 拟合，架构/种子/阈值是否只由 valid 决定，test 是否只做锁定后的评价。核对 B0 seed2029 checkpoint 与两次正式输出的 SHA。
3. 独立复算可取得的指标和数值恒等式：727条 test 的 Accuracy、Macro-F1、MAE、RMSE、Pearson、混淆矩阵；20条解释的概率和、Shapley 加和、IG 数值状态及异常样本保留。报告当前给出的 test 数字是 0.6588720770、0.5824303167、0.6475765522、0.8727662373、0.6603898510，**请以原始记录核实，而非引用这些数字作为证据**。
4. 审查 C2/T2/T3 是否真的支持原素材证据。C2/T2 汇总为1128条模态行、564个内容位置，T2 音频564唯一、视觉522唯一+1非唯一+41断链；T3 15个候选中9个媒体起点完整通过。检查18号勘误及06/13/16边界。禁止由局部通过推断所有音视频位置可对应秒数或关键帧。
5. 审查可复现性：源数据版本、代码、参数、环境、日志、manifest/SHA、样本覆盖和失败分支。指出任何只凭摘要无法独立重算的项目。

## 当前准备进展与公开分支限制

- 本地冻结工件核对：test 逐样本 CSV SHA 与审核记录一致；附件4 25 项输出 manifest 已逐文件核验；valid120 ZIP 含120个样本文件，内部129项 manifest 全部匹配，ZIP SHA 与审核记录一致。
- Attachment4 CSV 生成链：只读 exporter 从冻结 JSON 导出20行 CSV，SHA 与正式历史 CSV 完全相同。没有运行日志证明当时实际调用了哪份 exporter，因此历史调用路径仍标为 `UNVERIFIED_HISTORICAL_SOURCE`。
- 历史预 finalization 审计曾按旧字段口径报告分类/回归主要模态各18/20条；该计数已由当前 `q3_finalization_candidate_2026-09-26/results/final_evidence_coverage_20.csv` 按冻结最终语义与T4证据重新计算并取代。当前口径为 primary-reference 20/20、dominant-classification 20/20、dominant-regression 19/20；不定义 `primary_supporting_modality_reg`。
- 历史记录（audit bundle 发布前）：单独上传冻结 test CSV 的尝试曾被 GitHub 写入审核拒绝。随后用户要求将原始审核 ZIP 原样提交并公开可读展开；当前 CSV、Attachment4逐样本结果及 valid120展开材料均已纳入本分支。历史拒绝不再代表当前逐样本审核材料缺失。

## GitHub 证据的限制

公共分支现在公开了逐样本 test 预测、Attachment4结果、checkpoint/scaler、valid120展开结果及相应审计材料，外部审核者可以直接读取并复算已发布记录。原始约1 GB `aligned_50.pkl` 和附件4原始 aligned PKL 未公开，因此仅凭GitHub无法从原始输入端重新运行完整的 test/Attachment4推理。不得声称外部审核者已重跑原始模型，除非其实际获得并使用了这些输入。

候选结果允许表述为“当前冻结实现可以复现已冻结输出”；历史运行的确切调用来源仍为 `PARTIAL`，不能声称已证明历史 invocation provenance。缺少原始输入只限制端到端重跑，不影响读取当前公开CSV/JSON/图表以及按这些记录进行的独立数值检查。
## 请返回的格式

输出：①题意对照矩阵；②你实际读到并复算的文件和 Git commit；③P0/P1/P2问题及证据路径；④无法验证的项目和所需文件；⑤Q3 是否达到题意、外部4.6 Gate `PASS / PASS_WITH_LIMITATIONS / FAIL / BLOCKED`；⑥是否建议进入4.7。任何结论都应区分“独立重算”“核对摘要”“沿用他人报告”。

## 本地审核包的只读复核入口

若用户提供 `q3_external_audit_bundle.zip`，解压后可运行：

```bash
python 阶段4.2_Q3_v1_preflight/external_review_handoff_2026-09-26/external_4_6_test_audit.py --predictions 阶段4.2_Q3_v1_preflight/formal_test_evaluation_2026-09-26/test_predictions.csv --summary 阶段4.2_Q3_v1_preflight/formal_test_evaluation_2026-09-26/test_evaluation_summary.json --manifest 阶段4.2_Q3_v1_preflight/formal_test_evaluation_2026-09-26/output_manifest.json --out test_audit.json

python 阶段4.2_Q3_v1_preflight/external_review_handoff_2026-09-26/external_4_6_attachment4_audit.py --results 阶段4.2_Q3_v1_preflight/formal_attachment4_2026-09-26/results --out attachment4_audit
```

两个入口仅读取已冻结 CSV/JSON 和 SHA manifest，不加载 checkpoint，也不运行模型。不要把本地审核包里的历史内部交叉检查记录当作外部审核结论。

