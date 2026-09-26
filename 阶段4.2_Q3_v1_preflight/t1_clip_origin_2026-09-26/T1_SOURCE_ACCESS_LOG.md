# T1 来源媒体获取与执行记录

记录日期：2026-09-26。公开来源仅作本轮诊断，下载的媒体保存在私有本地目录，不提交到 GitHub。

## 可得性

- `YCEllKyaCrc`：YouTube oEmbed HTTP `200`，标题 `Charlie Chaplin - Final Speech from The Great Dictator (Clip)`；yt-dlp `2026.08.19` 元数据：`id=YCEllKyaCrc`、`duration=63`、`availability=public`、81种格式。`yt-dlp -f 251-7` 下载英语原音轨，`yt-dlp -f 134` 下载视频轨，两命令 exit code `0`。SHA-256、字节数在 `results/t1_frozen_contract.public.json`。
- `GAVpYuhMZAw`：YouTube oEmbed HTTP `404`；yt-dlp 返回 `This video is unavailable`，exit code `1`。未找到可核身份的来源长视频，未运行该样本的音频相关或画面起点分析。

## 执行命令概念

1. 在 Python 3.12 独立任务依赖目录安装 `yt-dlp==2026.8.19` 与 `av==18.1.0`；CSD 读取依赖为 `h5py==3.15.1`。任务私有媒体 SHA 固定后才运行波形比较。
2. `python freeze_t1.py`：保存事先判据与来源媒体 SHA。
3. `python validate_clip_origin.py`：使用已固定的 ffmpeg 解码器做全源音频归一化互相关及三个预选画面对照，生成 `results/t1_clip_origin_02.json`。
4. `python verify_frame_pts.py`：用 PyAV 解码实际 PTS，生成 `results/t1_frame_pts_02.json`。全101帧仅作为补充诊断；通过门仍按此前冻结的三帧判据。

`ffmpeg` 位于工作区 `work/tools/ffmpeg-9.0.2-essentials_build/ffmpeg-9.0.2-essentials_build/bin/`。来源媒体不是比赛附件，无权用其替换官方 MP4 或官方特征。
