# 步骤08：候选包副本验证

- 总结：PASS。
- 验证对象：候选包展开后的独立副本目录，不是原始生产目录。
- Python：`3.12.10`；解释器：`C:\Users\Fine\Documents\Codex\2026-09-23\https-github-com-hahayyf666-netizen-shuxuejianmo\work\.venv_q1\Scripts\python.exe`。
- 候选ZIP：`outputs/q1/v1_delivery/package/q1_v1_candidate.zip`，23,623,432 bytes；SHA-256 `d1b3bb587bd96b4094b555eeab1add4eed048c316ee84fbb6d8a3456fa39d7a2`。
- ZIP CRC：PASS；NPZ文件：100/100。
- 独立结构验证退出码：0；输出：`PASS: 100 samples; modes={'UNCERTAIN_REVIEW': 88, 'TRI_MODAL_WORD_VALID': 5, 'AV_VALID_TEXT_UNALIGNED': 5, 'TRI_MODAL_WITH_AUDIO_CONTENT_INVALID': 2}; feature_bytes=17295166`
- 读取器样本测试退出码：0；官方文本：`-s9qJ7ATP7w$_$6` / `But I just kept going`。

完整命令、stdout/stderr和环境记录在同目录的`step08_package_copy_validation.json`。
