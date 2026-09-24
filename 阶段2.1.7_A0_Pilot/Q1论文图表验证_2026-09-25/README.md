# Q1 论文辅助验证图

本目录保存基于正式 Q1 v1 特征文件生成的词级三模态特征热图及其完整证据链。该图没有重新提取或修改任何特征，也没有进入 2.2。

## 文件

- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.svg`：论文优先使用的矢量版本；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.pdf`：PDF矢量容器版本；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.png`：600 dpi、4251×3661像素的审阅和Word插图版本；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6_data.csv`：图中全部4,610个原始值与标准分数，仅保留在本地审计目录，不上传GitHub；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6_manifest.json`：源NPZ、变换公式、软件版本、哈希与机器检查；
- `generate_trimodal_feature_heatmap.py`：确定性生成脚本；
- `图表说明与论文写作用法.md`：图注、正文建议和解释边界。

## 证据定位

- 样本：`-s9qJ7ATP7w$_$6`；
- 官方文本：`But I just kept going`；
- 对齐状态：`TRI_MODAL_WORD_VALID / word_valid`；
- 正式源特征：`outputs/q1/v1_delivery/features/-s9qJ7ATP7w___6.npz`；
- 源NPZ SHA-256：`ec142e3a82796729b2cd8526cb7becff2daeeebe288b696e5f1443abee71eb12`。

## 图的证明范围

该图用于验证：同一组官方词及其可信时间区间上，正式输出确实形成了`5×768`文本、`5×50`语音和`5×104`视觉特征矩阵；三块热图共享完全相同的词列，因此可以直观看到词位组织的一致性与各模态特征的局部变化。

该图不证明情感分类性能，不把颜色解释为情感强度，也不比较三种模态的原始数值大小。

GitHub公开同步包包含图形、脚本、manifest和论文用法说明。完整逐特征CSV因包含正式内部特征数值，仅作为本地复核底稿保存。

## 绘图质量流程参考

绘图完整性与可复现检查参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
