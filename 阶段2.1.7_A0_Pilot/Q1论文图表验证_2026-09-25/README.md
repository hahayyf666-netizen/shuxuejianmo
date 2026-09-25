# Q1 论文图表验证

本目录保存同一典型样本的**三模态时序对应主图**与**三模态特征热图**。两图均基于正式 Q1 v1 产物；本次补发主图没有重新提取或修改特征，也没有进入 2.2。

## 主图：三模态时序对应

- [`typical_correspondence_s9qJ7ATP7w_clip6.png`](typical_correspondence_s9qJ7ATP7w_clip6.png)：论文主图，3750×2160 像素；从已冻结的 `q1_v1_candidate.zip` 原样提取，SHA-256 为 `d4dd9ddf58eaf95571aceb743f36d2229c0df5a2f27b84503d5c575f07ec7fb0`。
- [`typical_correspondence_s9qJ7ATP7w_clip6_standalone.svg`](typical_correspondence_s9qJ7ATP7w_clip6_standalone.svg)：便于论文排版的独立 SVG；只将原 SVG 所引用的 6 张栅格图嵌入文件，未改动图形内容。原 SVG 使用生成机器上的绝对路径，不能直接作为可移植文件发布。
- [`typical_correspondence_s9qJ7ATP7w_clip6.json`](typical_correspondence_s9qJ7ATP7w_clip6.json)：原始图证据，含样本、源 MP4/NPZ 哈希、逐词区间、音频窗与视频帧索引、center 归词复算结果；其中的 SVG 路径和 SHA 指向压缩包内原 SVG，不指向上述独立 SVG。
- [`figure_layout_check.json`](figure_layout_check.json)：原始版面检查，结论为 `PASS`。
- [`make_standalone_correspondence_svg.py`](make_standalone_correspondence_svg.py)：从冻结 ZIP 重建独立 SVG 的脚本。
- [`verify_main_figure_publication.py`](verify_main_figure_publication.py)：用冻结 ZIP 复核哈希、五词归位证据、SVG 可移植性和 PNG 尺寸。
- [`main_figure_publication_manifest.json`](main_figure_publication_manifest.json)：发布文件与冻结压缩包的哈希对应关系。

主图按共享 presentation timeline 展示官方词区间、实际解码音频波形、音频特征窗中心、视频帧 PTS 以及每词的一张示例原始帧。着色点满足 `start <= center/PTS < end`，其余灰点保留以显示原生时间序列。展示用视频帧的人脸区域已马赛克；原 MP4 和正式特征文件未修改。

## 辅助图：三模态特征热图

- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.svg`：论文优先使用的矢量版本；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.pdf`：PDF矢量容器版本；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6.png`：600 dpi、4251×3661像素的审阅和Word插图版本；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6_data.csv`：图中全部4,610个原始值与标准分数，仅保留在本地审计目录，不上传GitHub；
- `q1_trimodal_feature_heatmap_s9qJ7ATP7w_clip6_manifest.json`：源NPZ、变换公式、软件版本、哈希与机器检查；
- `generate_trimodal_feature_heatmap.py`：确定性生成脚本；
- `图表说明与论文写作用法.md`：两图的图注、正文建议和解释边界。

## 证据定位

- 样本：`-s9qJ7ATP7w$_$6`；
- 官方文本：`But I just kept going`；
- 对齐状态：`TRI_MODAL_WORD_VALID / word_valid`；
- 正式源特征：`outputs/q1/v1_delivery/features/-s9qJ7ATP7w___6.npz`；
- 源NPZ SHA-256：`ec142e3a82796729b2cd8526cb7becff2daeeebe288b696e5f1443abee71eb12`。

## 图的证明范围

主图用于核验：这一条可信词级样本的五个官方词，在实际音轨和画面的共同时间轴上如何选择音频窗与视频帧。热图用于验证：同一组词及其可信时间区间上，正式输出形成了 `5×768` 文本、`5×50` 语音和 `5×104` 视觉特征矩阵；三块热图共享完全相同的词列。

两图仅证明典型可信词级分支的时序规则和输出结构，**不代表全部 100 条均完成词级对齐**，也不证明情感分类性能。热图颜色不是情感强度，三种模态的原始数值大小不能据此比较。

GitHub公开同步包包含图形、脚本、manifest和论文用法说明。完整逐特征CSV因包含正式内部特征数值，仅作为本地复核底稿保存。

## 绘图质量流程参考

绘图完整性与可复现检查参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
