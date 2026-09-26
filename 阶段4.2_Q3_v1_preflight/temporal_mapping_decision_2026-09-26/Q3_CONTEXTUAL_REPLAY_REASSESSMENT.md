# Q3 原素材回看映射复核：对上次决策的修订

日期：2026-09-26。范围：只读复核原题、附件4的 C-2/C-4 审计结果，以及“aligned position → reconstructed time”的新建议。未运行 forced alignment、附件4正式推理或新训练；未修改模型、Shapley、IG、现行解释合同和历史审计工件。

## 1. 原题与上次判断

原题问题3要求关键证据可对应原始文本片段、语音时段**或**视觉关键帧；附件4原视频用于将关键证据时间位置回看至原素材。题目又称 `aligned_50` 三模态按最多50个“相互对应的序列位置”组织。这些话支持探索**可复核的原素材回看导航**，并未要求法证式恢复官方提取器使用的每一个窗或帧。

因此，上次报告 `Q3_TEMPORAL_MAPPING_INDEPENDENT_DECISION.md` 中拒绝把 forced alignment + PTS 命名为 `verified_official_feature_time` 的结论保留；但其 `NO_NEW_TEMPORAL_MAPPING_RUN` 如果覆盖一切独立的**上下文回看诊断**，范围过宽。本文件修订这一点，不改写上次报告。

## 2. 新建议遗漏的实测反证

新建议的关键等式是 `text[i]` 的说话时间 = `audio[i]` / `vision[i]` 的原素材时间。题目中的“相互对应”尚不足以证明这个**精确到同一词时间窗**的等式。C-2 的真实行匹配给出了很强的警示：

| C-2 只读事实 | 数量 |
|---|---:|
| 对齐 audio 内容行在同样本未对齐 audio 中有唯一、逐维完全相等的来源行 | 564/564 |
| 对齐 vision 内容行同样有唯一来源行 | 523/564 |
| audio 来源行的集合恰好构成未对齐有效序列的末尾连续段 | 18/20 样本 |
| vision 来源行的集合恰好构成末尾连续段 | 16/20 样本；另有 13、16 两条全部视觉行未匹配 |

例如附件4样本 `02` 的第1个文本内容位置是原文 **“We”**，整条原文有19个内容 token 位置，视频呈现时长约3.493秒；对齐音频位置1对应未对齐音频第52行（从0起，共66行），对齐视觉位置1对应未对齐视觉第36行（共50行）。其19个对齐音频位置只落在来源行52–65，视觉位置只落在36–49，多个 token 复用同一来源行。按 C-4 的20 Hz/15 Hz **诊断代理网格**估算，这些来源行位于视频尾部约0.7–1.0秒；代理网格不是已验证的行时间戳，故不能把估算秒数当结论，但这种末尾聚集与“逐词覆盖整段话”的假设明显紧张。

样本 `01` 也类似：22个文本内容位置对应的音频来源行158–178/179、视觉来源行114–134/135；视频约9秒。样本 `07`、`18` 因序列上限等原因没有形成完整末尾集合，样本 `13`、`16` 则有独立视觉例外。以上模式说明：**先把 `text[i]` 的词时间直接赋给 `audio[i]`/`vision[i]`，可能把模型高分行定位到错误片段。** 它还不是严格证明所有样本都错位，因为官方行到原始秒/帧尚未被直接核验。

另一个独立风险是 forced alignment 以给定文本为条件：单调、不越界、无零时长只检查形式，不证明说话内容与原文一致。Q1 已实际发现官方文本与音轨不一致的样本。因此“词序正确 + PTS正确”不能单独通过映射门。

## 3. 分开两个可交付层级

1. **官方模型归因**：B0 的 Shapley 模态贡献与 conditional IG 官方序列位置仍有效；附件4文本 `text` 行到 WordPiece/原文字符范围已由 C-4 在564/564内容行验证。
2. **独立原素材上下文回看**：可从原 MP4 的 presentation 音轨核对实际说话内容，用 forced alignment 产生候选词时间，再按解码 PTS 取对应画面。只要这一层未与官方音视频特征行独立核对，它最多标为 `context_replay_candidate`，**不能写成 audio/vision IG 行的关键秒数或关键帧**。这层可以帮助人工回看和给论文提供素材上下文，但不能充当模型位置归因的验证。

需要分别记录 `text_row_to_char_status`、`audio_feature_row_to_media_status`、`vision_feature_row_to_media_status` 和 `independent_context_replay_status`，不要用一个 `mapping_status` 掩盖证据链的断点。现行正式合同仍保持 audio/vision `index_only`。

## 4. 若继续尝试，先做短而可证伪的 T0 门

不采纳“直接对3–5条强制对齐，通过后映射20条”的原方案。若为补足题目回看证据而单独授权一个小范围实验，应先预选涵盖正常、超长、视觉异常的样本，优先用现有 C-2 来源索引检查下列独立连接：

1. 核媒体 SHA、编辑列表和 audio/video 共享 presentation PTS；核 `raw_text` 与实际语音内容。内容明显不一致或静音时，对应词时间不出图。
2. 不用文本词时间去**定义**音视频官方行时间。先独立建立“未对齐 audio 来源行 → 音轨时间”和“未对齐 vision 来源行 → 视频 PTS”的候选链接，预设容差及错位/倒序对照；仅凭20/15 Hz 时长比例、单调性或 CKA 不算通过。
3. 将经内容审核的文本词时间，与上述**独立**的来源行候选时间比较。如果 `text[i]` 与 `audio[i]` / `vision[i]` 系统性错位，立刻停止词时间继承；记录失败样本与误差，不做逐样本调偏移以追求通过。
4. 若个别行通过，可仅对这些行报告 `reconstructed_context_time`、误差/不确定度和 `official_extractor_provenance=unverified`；未通过行继续 `index_only`。视觉 `13/16` 不能由音频成功自动升级。

这只是**候选门的设计**，本次没有执行，也没有预先宣称它会通过。20条正式解释及协议字段若要改变，须待 T0 真实结果和阶段审核。

## 5. 决策与当前阶段

**决策：拒绝把 `text[i]` 的 forced-alignment 时间直接赋给 `audio[i]` / `vision[i]`；撤回上次对一切上下文回看诊断的笼统禁止。** 目前最有效的下一步是先检查同位置的时间语义，只有独立链接通过的行才讨论原素材回看。现有 B0、valid120 解释验证不因此作废；Q3 仍在 REVIEW GATE，现行 audio/vision `index_only` 不变。

### 核对来源

- 原题 DOCX：`复杂场景下多模态情感识别的数学建模与算法设计.docx`，SHA-256 `38cec978723bff36ab945fa98f6454558bb9084e2cd01d814dd6e38ab4b4c336`；问题3、附件4说明、表1。
- C-2：`stage_c2_provenance/results/pair_inventory_20.csv`、`aligned_to_unaligned_row_matches_20.csv`、`media_pts_20.csv` 及 `pairwise_source_summary.json`。
- C-4：`stage_c4_feature_reconstruction/results/text_row_trace_564.csv`、`Q3_FEATURE_RECONSTRUCTION_REPORT.md`。
- 上次决策：`temporal_mapping_decision_2026-09-26/Q3_TEMPORAL_MAPPING_INDEPENDENT_DECISION.md`。
