# Q1第一版需求追踪表

范围：对照题目分析中的Q1交付要求、用户提供的官方答疑口径及第一版执行计划；机器PASS仅表示对应合同与产物通过，不扩展为未验证的内容准确率。

| 要求 | 证据 | 结果 | 状态与边界 |
|---|---|---|---|
| 赛题与答疑要求的输入保持原样 | outputs/q1/v1_delivery/01_全量特征合同.md; metadata/official_answer_evidence.json; input_manifest.csv; manifest.csv | 官方text列逐字保留；A/V来自对应MP4；100个原始样本均在manifest中；来源SHA可追溯。 | PASS；ASR只作诊断，不替代官方文本。 |
| 给出文本、音频、视觉特征方案与参数 | 01_全量特征合同.md; 02_字段与文件规范.md; config_snapshot.json; environment.json; requirements-lock.txt; metadata/model_assets.json | RoBERTa 768维、eGeMAPS 25维、MediaPipe blendshape 52维；可用词级派生维度50/104；版本、配置、权重身份与SHA均有记录。 | PASS；本阶段不训练或报告情感分类性能。 |
| 保存可信时间对应与异常分支 | 01_全量特征合同.md; audit_100.csv; full100 audit alignment; prefreeze_transcript_audio_2026-09-24; additional_controls_2026-09-24; outputs/q1/pilot/alignment | A/V使用共享presentation PTS；词区间半开；只对内容对应证据支持者写词时间，中心点规则及底层索引可复算；错配、静音、无脸、证据不足分别编码。 | PASS_WITH_LIMITS；仅5条进入可信词级映射；5条文本错配不造词时间；2条确认无有效人声仍保留声学特征；88条证据不足以断言词级映射，按clip-only输出且不要求用户逐条复核。 |
| 附件1全量100条各有结果文件及行表 | features/*.npz; manifest.csv; results_100.csv; reports/step06_full_output_audit.md | 100个唯一样本、100个NPZ、100行manifest、100行results；NPZ均可allow_pickle=False读取，manifest与results逐字节一致。 | PASS；100条处理成功不等于100条均获得词级跨模态时间映射。 |
| 有效长度、mask和原素材覆盖关系可检查 | 02_字段与文件规范.md; code/feature_reader.py; code/validate_q1_v1.py; reports/automatic_validation.json | 各模态保留独立变长；mask、PTS、源索引、词级有效性分开保存；无脸与静音不被写作模态不存在。 | PASS；24条未取得有效面部blendshape，仍保留视频帧和位置；3个词区间没有可选视觉帧并显式mask。 |
| 至少一个典型时间对应验证 | examples/典型对应验证.md; examples/typical_correspondence_s9qJ7ATP7w_clip6.png; reports/word_level_observation_gaps.csv | 展示“But I just kept going”5词区间、A/V中心索引独立复算、PTS帧和一静音/错配/无脸/零时长异常实例。 | PASS_WITH_LIMITS；人工内容依据为用户先前对两种edit-list试听的听辨：两段均听到该短语，忽略edit-list版本前面多一句；不据此声称整段只有该句或边界达到绝对真值。 |
| 版本、流程、读取与复现说明可执行 | README.md; code/run_q1_v1.py; code/validate_q1_v1.py; code/read_q1_sample.py; reports/reproduction_check.json; reports/step08_package_copy_validation.json | CLI --help与读取器已运行；固定9条跨分支复现9/9、字段数组在rtol=atol=1e-6下相符；候选包副本全量验证100/100。 | PASS_WITH_LIMITS；没有宣称对100条进行了第二轮完整特征生产；复现集为预先固定的9条分支样本。 |
| A0历史pilot保护与全题附件预算 | metadata/protected_a0_baseline.json; reports/step06_full_output_audit.md; package/q1_v1_candidate.zip; package/size_report.json | A0保护清单48/48 SHA一致；Q1候选ZIP实测字节数及剩余预算已记录。 | PASS_WITH_LIMITS；Q1包在50,000,000 bytes口径下合规；Q2/Q3加入后的全题最终总包尚未生成，最终总量需后续复核。 |

## 总体判断

Q1 v1已经有完整100条特征文件和结果表，且异常数据保留为明确的模态有效性/对应状态。它完成的是按可信粒度提供特征与时间关系：只有5条具有可信词级text-A/V时间映射。其余样本仍保留官方词特征和原生音视频序列；不能据100/100工程PASS推断所有文本与人声都匹配。

此表是步骤09需求核对，不代表2.2改进、2.3红队冻结或全题附件总量验收已经完成。

