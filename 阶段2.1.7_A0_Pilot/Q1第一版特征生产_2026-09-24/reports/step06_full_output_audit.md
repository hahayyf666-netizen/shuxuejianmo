# 步骤06：全量产物与旧A0保护核查

- 总结：PASS。
- 输入、manifest、结果表、NPZ：100 / 100 / 100 / 100。
- 验证器：100/100 PASS；失败 0。
- NPZ体积：17295166 bytes。
- 对齐模式：`{"UNCERTAIN_REVIEW": 88, "TRI_MODAL_WORD_VALID": 5, "AV_VALID_TEXT_UNALIGNED": 5, "TRI_MODAL_WITH_AUDIO_CONTENT_INVALID": 2}`。
- text/audio关系：`{"not_asserted": 88, "confirmed_match": 5, "confirmed_mismatch": 5, "no_speech": 2}`。
- audio_speech_valid：`{"-1": 88, "1": 10, "0": 2}`。
- A0基准：48/48项SHA-256一致。

各项检查：

- PASS：`manifest_rows_100`
- PASS：`input_rows_100`
- PASS：`results_rows_100`
- PASS：`unique_manifest_keys`
- PASS：`input_manifest_keyset_matches`
- PASS：`results_identical_manifest`
- PASS：`feature_files_100`
- PASS：`no_missing_npz`
- PASS：`all_npz_hash_and_readable`
- PASS：`no_unregistered_npz`
- PASS：`validator_100_pass`
- PASS：`all_a0_baseline_hashes_match`

解释边界：`UNCERTAIN_REVIEW` 表示当前机器证据不足以声明官方词与音频存在可靠逐词对应，因此仅使用片段级A/V时间表征；此标签不是“人工已复核通过”，也不代表样本未处理。静音和文本错配均按独立字段表达。
