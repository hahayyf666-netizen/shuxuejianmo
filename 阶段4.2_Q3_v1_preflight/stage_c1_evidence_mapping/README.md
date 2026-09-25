# Stage C-1：Evidence Mapping专项

本专项只审计证据映射，不修改模型、训练参数、预测接口或 Shapley/IG 定义，也不运行 Attachment4 预测。

诊断问题：原 preflight 中 285 条代表性映射为什么全部是 `index_only`？此状态是 fail-closed 输出，不应误读为 285 次实际映射失败。必须区分：

- **B / 当前预检路径未执行映射**：`run_mapping_precheck.py` 对每个模态和内容位置直接调用 `index_only()`，并把 285 记为 `index_only`。它没有调用 tokenizer offset、forced aligner、PTS 映射或相应失败诊断。这是本次全部 285 行同状态的直接原因；不能据此说 285 次映射都失败。
- **A / PKL显式来源信息缺口**：附件4 20 个 PKL 的观察到的键为 `audio,id,raw_text,text,text_bert,vision`，未见逐行时间、PTS、帧号或特征来源字段。这使 PKL 本身不足以证明特征行到原始字符/时间/帧的关系；但**尚未证明该关系无法从上游提取代码、配置或来源文档恢复**。

所以根因是 **B 为直接原因，A 为当前交付文件的剩余证据缺口**。本轮没有用原视频重算音视频特征，也没有运行 CTC forced alignment；在找回并核验实际官方提取链前，Audio/Vision 不能据原始视频存在就判定映射通过。当前映射门仍为 BLOCKED，保持 fail-closed。

## 复现

使用官方 `google-bert/bert-base-uncased` 固定快照的 `vocab.txt`（revision `b96743c503420c0858ad23fca994e670844c6c05`，SHA-256 见报告），Transformers `4.57.6`，用 `BertTokenizerFast` 对附件4全部20条 raw_text 运行 lower-case、max_length=50、truncation、padding=50、offset mapping。脚本参数化接收词表与附件4目录；不下载或使用模型权重，不运行预测。

```powershell
python .\probe_tokenizer_mapping.py `
  --vocab 'C:\path\to\vocab.txt' `
  --attachment4-aligned 'D:\path\to\附件4\对齐版本' `
  --out .\results
```

`text_tokenizer_replay_20.csv` 对全部 20 条附件4样本重放 token IDs、attention mask、token type IDs，并导出字符 offset。3 条预定代表样本 01/13/20 的 95 个内容位置均可获得精确 `raw_text -> text_bert` 字符范围候选。但这仍不会证明 Q3 使用的连续 768 维 `text` 特征行来源，故不能标 `verified_text`。

音视频的 `index_only` 不表示素材不存在：20/20 个原视频配对文件存在。它表示当前 PKL 未提供逐行音频时间或视频 PTS/帧来源，而先前 precheck 未尝试时间重建或特征行支持验证。MMSA 官方 schema 描述 `text`、`text_bert`、`audio`、`vision` 与长度字段；MMSA-FET 文档说明其可选 Wav2vec CTC aligner 能生成词时间并把音视频对齐到文本，但这些公开说明不能单独证明本比赛附件4特征文件由该确切版本、配置与流程生成。见 [MMSA 数据格式说明](https://github.com/thuiar/MMSA/blob/master/README.md#2-datasets) 与 [MMSA-FET 对齐器说明](https://github.com/thuiar/MMSA-FET)。
