# 附件1全量100条模态有效性与跨模态对应关系审计规则

状态：Q1 REVIEW GATE。本目录只记录审计证据，不修改 `label-100.xlsx`、原始 MP4、冻结的三条 A0 pilot、2.1.7 或任何模型参数；不进行最终100条三模态特征生产。

## 输入与身份

- 总体为 Stage1 清单与 `label-100.xlsx` 共同列出的100个唯一 `sample_key`。逐条检查 MP4 文件身份 SHA-256；official_text 逐字取工作簿 `text` 列。ASR 仅为无提示的复核线索，不替代 official_text。
- 音频和画面只取同一原始 MP4 的实际解码内容。采用当前 shared presentation timeline 和解码 PTS；不按 `frame_index/fps` 重建时间，不切换 edit-list 语义。
- 已有100条无提示 ASR 筛查作为内容差异候选来源；其 high/moderate/close/empty 标记均不能独立证明官方文本匹配与否。既有逐条人工结论按可追溯样本键记录，未获人工结论的条目统一 `human_review_status=pending`。

`official_text` 为官方工作簿原文；`asr_text` 为原MP4音轨的无提示机器转写，仅供诊断。机器转写有误识别或幻觉的可能，不作为新文本真值。字幕、讲话和画面的语义匹配并未由解码成功或PTS检查自动证明。

## 字段定义

表内的二值字段使用 `1`、`0`；机器或人工证据不足时使用 `unknown`。`audio_present=1` 表示原 MP4 有音轨，数字静音仍为1。`video_present=1` 表示有视频轨，未检测到人脸仍为1。`text_present=1` 只表示 official text 非空；`text_content_valid=1` 表示该文本含可用词语，不代表它与原视频相符。

`audio_speech_valid=0` 只在解码音轨逐采样全零，或经人工明确确认没有人声时填写。`audio_speech_valid=1` 需要已有人工确认说话内容或同等强证据。其余非零音频写 `unknown`；音频能量、ASR 文字及概率只保存在证据列。这个字段不改变现有 pilot 的 `audio_valid`：后者表示词窗能否聚合音频观测；未来如需区分可另议 `audio_observation_valid`，本轮不改旧字段。

`face_feature_valid=1` 表示已冻结 MediaPipe FaceLandmarker 在至少一帧返回一组可用的52维 blendshape 类别；全帧都没有可用输出则为0。它是**该提取器的覆盖状态**，不是“画面里没有脸”的事实判定。对本题当前面部视觉特征，`vision_feature_valid` 与这个覆盖状态同口径；视频模态存在性单独记在 `video_present`。审计只记录每帧检测数和覆盖率，不保存或聚合任何新的 blendshape 向量。

`audio_visual_time_valid=1` 要求两轨均可解码，解码后的 PTS 可读、单调，音视频起点可用当前共享 `t0` 解释，且呈现覆盖区间有正时长重叠。失败或无法判定时记录0/unknown及具体证据，不自动修复。静音和无脸不影响这一字段。

`text_audio_correspondence` 取 `confirmed_match`、`confirmed_mismatch`、`no_speech` 或 `unknown`。前两项仅来自已明确的用户听辨，不从 ASR 距离、低对齐概率或零时长词自动推出。`text_av_time_mapping_status` 取 `word_valid`、`word_partial`、`clip_only`、`unavailable`、`unverified`、`timeline_anomaly`。`clip_only` 表示原 MP4 的整体时间区间明确，但官方词不能定位；不得制造词时间戳。

`human_review_required=1` 是原机器候选清单的未确认标志：该样本若要取得“人工确认”的结论，仍需相应证据。按用户后续指示，它**不代表当前要求用户逐条听辨，也不作为机器审计完成的门槛**。88条继续pending，主表不伪造人工状态；汇总另以 `mandatory_user_listening_count=0` 明确本轮没有指派人工听辨任务。

`TRI_MODAL_WORD_VALID` 当前5条的证据层级是“已有人工整句内容匹配结论＋冻结对齐器的词区间结构检查通过”，并未测量逐词边界误差，不应由该类别推导毫秒级准确或所有词边界已被人工验证。`audio_visual_time_valid=1` 证明本次解码时间坐标和覆盖关系可解释，不证明口型同步或画面语义与文本一致。

## 对齐模式判定顺序

1. 原始素材身份、解码或共享 A/V 时间轴不能可靠确定：`EXTRACTION_OR_TIMELINE_ANOMALY`，只保存证据。
2. A/V 时间轴可信且实际音轨逐采样静音：`TRI_MODAL_WITH_AUDIO_CONTENT_INVALID`；保留音频观测，`audio_speech_valid=0`，只保留 clip 级时间位置，不宣称官方词级对应。
3. A/V 时间轴可信、人工确认官方文字与实际讲话明显不对应：`AV_VALID_TEXT_UNALIGNED`，`text_av_time_mapping_status=unavailable`。
4. 人工确认文本与讲话基本对应，且使用冻结 stable-ts 官方文字强制对齐后全部官方词具有合法、单调、覆盖内的正时长区间：`TRI_MODAL_WORD_VALID`。此模式只判断词时间映射；面部特征覆盖另看 `face_feature_valid`。
5. 其余情况：`UNCERTAIN_REVIEW`。机器概率、ASR 距离或结构合法性不准自动提升为内容确认。所有未获明确人工结论的条目均为 `human_review_status=pending`。

已有零时长词、mapping failure、非单调/越界和概率仅作为诊断统计。数字静音样本不因强制对齐失败而记为整个样本 failure；官方文本与讲话不符先记录 correspondence anomaly。只有原始样本身份、数据来源、A/V 共同时间轴无法可靠确定，或继续处理必须修改官方数据或核心语义，才报告计划级 HARD STOP。

## 审计边界与输出

- `audit_100.csv` 一行一条；`audit_summary.json` 给出所有模式与字段计数及未确认范围；`machine_review_candidates.csv` 记录需人工复核的样本和优先级。
- `media/` 记录100条视频帧 PTS 和 MediaPipe 面部覆盖诊断；`alignment/` 记录冻结 stable-ts 的逐条诊断统计及证据来源。`logs/` 保存命令、版本、退出码和运行输出。已有人工作答仅作为历史证据写入本轮汇总，不反写旧工件。
- 本轮完成机器审计与候选清单即停止。没有逐条人工复核的条目保持待审；不据此修改2.1.7、运行MFA或进入2.2。

## 已有证据补充与阶段口径

两条数字静音样本已使用冻结openSMILE配置完成独立的声学可行性探针，见相邻诊断目录 `../exception_handling_feasibility_2026-09-24/静音声学最小探针.json`。它们分别产生601×25、566×25的有限LLD值；这证明有真实声学观测窗口可供提取，未改变 `audio_speech_valid=0`，也未生成对应官方词的可信时间边界。不得据此将尚未提取的其他样本声学窗口标为已验证。

本轮机器审计闭合与Q1第一版全量求解闭合分开记录。按用户最新要求，先完成Q1第一版100条特征及配套交付，再进入2.2检查与改进；目前仍在REVIEW GATE／第一版准备。Notion中的旧统一词级输出及文本错配HARD STOP条款与用户提供的最新答疑处理口径存在待同步之处，本轮只记录差异，未直接编辑Notion。
