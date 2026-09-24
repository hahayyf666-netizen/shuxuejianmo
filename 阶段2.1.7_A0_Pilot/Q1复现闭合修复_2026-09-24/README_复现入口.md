# Q1 多模态特征 v1.0

> **2026-09-24 Q1 v1独立P2只读验收：PASS。** 两项P1复现缺口已修复；迁移干净CPU环境、100条全量结构审计、9条冻结分支复现和5条词级trace回放均通过。P2独立检查未发现阻止第一版功能闭合的缺口。88条样本不主张词级文本对应，ASR只作为机器筛查。详见[Q1第一版闭合验收报告](Q1复现闭合修复_2026-09-24/Q1第一版闭合验收报告.md)和[独立P2回执](Q1复现闭合修复_2026-09-24/reports/P2独立只读验收_2026-09-24.md)。此前NEEDS_REVISION审计留作历史证据。继续保持2.1.7 REVIEW GATE；本README不表示已进入2.2或Q1最终冻结。

本目录保存 Q1 第一版特征、100行结果表、合同、代码、验证材料和可复核示例。该版本用于完成问题1的特征提取与可用时间对应；它没有训练或评估情感分类器。

## 结果入口

- `manifest.csv` 与 `results_100.csv`：100条样本的权威行表；当前两个文件逐字节相同。
- `features/`：100个单样本NPZ。每个文件使用 `q1-feature-v1.0` schema，不使用pickle对象数组。
- `reports/automatic_validation.md`：全量独立结构验证结果。
- `reports/Q1方法与结果说明.md`：特征定义、时间语义、模式计数及结论边界。
- `examples/典型对应验证.md`：内容证据、5个词区间、中心点选择索引和异常示例。
- `code/feature_reader.py` 与 `code/read_q1_sample.py`：统一读取接口和命令行摘要。

本次正式运行的最终结果：100/100个NPZ和100/100行manifest通过；特征文件合计17,295,166 bytes。模式计数及静音/无脸/错配状态见方法说明。`UNCERTAIN_REVIEW`表示文本到音视频的词级对应不作断言，仍完整输出官方文本特征和原生A/V序列；该模式不构成逐条用户听辨任务。

## 输入目录和前置证据

运行入口依赖以下输入；移动仓库后可用命令行参数指定新位置。

1. `--input-root`：赛题附件1目录，根下有 `label-100.xlsx` 和 `<video_id>/<clip_id>.mp4`。标签表的 `text` 列为唯一官方文本源。
2. `--audit-root`：100条全量对应审计，需有`audit_100.csv`、`audit_summary.json`、`audit_rules.md`、`alignment/`逐条对齐诊断和`media/`逐条原始媒体解码/PTS证据。此路径必须位于解压后的项目根目录内，因为`prepare`会记录其项目内相对路径；本包默认路径为`outputs/q1/diagnostics/full100_correspondence_audit`。
3. 仓库内 `outputs/q1/pilot_selection_metadata.csv`：冻结的Stage1样本元数据。
4. 仓库内 `outputs/q1/pilot/` 与 `outputs/q1/pilot_samples.json`：A0旧工件只读保护基准。`prepare`逐项校验48个旧文件哈希。
5. `--asset-root`：默认 `work/model_assets_a0`，目录结构和资产SHA见 `metadata/model_assets.json` 及 `work/model_assets_a0/asset_manifest.json`。

所有原始MP4与标签表都通过SHA-256在输入清单中锁定。不要把附件中的官方文本替换为ASR结果，也不要修改A0 pilot或全量审计来绕过哈希检查。

## 环境和固定资产

正式运行环境见 `environment.json`、便携安装说明和 `metadata/model_assets.json`：Python 3.12.10，CPU；torch/torchaudio 2.8.0，stable-ts 2.19.1，openai-whisper 20250625，Transformers 4.57.6，openSMILE 2.6.0，MediaPipe 0.10.35，PyAV 18.0.0。正式特征生产不运行Whisper转写或新forced-alignment，也不需要GPU。`FaceLandmarkerOptions`的有效默认confidence阈值（检测、存在性、跟踪）均为0.5；这些值由固定MediaPipe版本的API默认提供，逐项记录见`Q1复现闭合修复_2026-09-24/metadata/effective_mediapipe_parameters.json`。音频start/end是openSMILE返回的时间标签，center用于归词，不把它们解释为各维度共同的物理窗支持。

- 文本模型：`FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`。权重SHA-256：`5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`。所有模型文件的SHA在 `metadata/model_assets.json`。
- 面部模型：MediaPipe `face_landmarker.task`，来源地址记录于 `work/model_assets_a0/asset_manifest.json`；文件SHA-256：`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。
- openSMILE 使用锁定版本自带的 eGeMAPSv02 配置；配置文件SHA-256：`ef451953badced2ed112ba4ffb997f9d2a7e7c444a09e8092665eadd9f24109e`。
- checkpoint和附件不打入小型候选包；如重新下载，必须把文件放入manifest规定的相对目录，并在运行前校验SHA。哈希不一致时不要运行特征生产。

在Windows PowerShell中，从包含`outputs/q1/v1_delivery/code/run_q1_v1.py`的仓库根目录运行。先下载并逐项校验外部模型权重，再创建Python 3.12.10 CPU环境。模型总量约650 MB；它们不放进候选ZIP。

```powershell
$projectRoot = (Get-Location).Path
$assetRoot = Join-Path $projectRoot "work\model_assets_a0"
$whisperRoot = Join-Path $projectRoot "work\whisper_cache_2_19_1"
$venvPath = Join-Path $env:TEMP "q1env_q1_closed"
$installReportRoot = Join-Path $env:TEMP "q1install_reports"
$basePython = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
& "outputs\q1\v1_delivery\Q1复现闭合修复_2026-09-24\environment\download_q1_assets.ps1" -AssetRoot $assetRoot -WhisperRoot $whisperRoot
& "outputs\q1\v1_delivery\Q1复现闭合修复_2026-09-24\environment\install_repro.ps1" -BasePython $basePython -VenvPath $venvPath -AssetRoot $assetRoot -WhisperRoot $whisperRoot -ReportRoot $installReportRoot
$pyExe = Join-Path $venvPath "Scripts\python.exe"
```

将虚拟环境和安装报告放在系统临时目录的短路径，避免Windows对深目录安装包的路径长度限制。安装输出、pip freeze、安装来源报告和校验回执均写入`$installReportRoot`。

资产脚本固定下载`FacebookAI/roberta-base`的指定revision、MediaPipe face landmarker和OpenAI Whisper `base.en`权重，并将下载文件的SHA-256与`metadata/model_assets.json`中的身份一致地校验。运行环境验收保存`pip check`、完整pip freeze、CPU wheel安装报告和PyPI源码分发报告。`requirements-lock.txt`仅转引版本清单；canonical安装入口是PowerShell脚本，因为它还核对PyTorch CPU wheel来源及两个源码包SHA。`prepare`记录的pip快照写入`outputs/q1/v1_delivery/reports/pip_freeze_snapshot.txt`，不会覆盖安装锁或来源报告。

## 运行与续跑

示例环境变量仅在当前PowerShell会话内使用：

```powershell
$pyExe = "work\.venv_q1\Scripts\python.exe"
$inputRoot = "D:\path\to\附件1"
$auditRoot = "outputs\q1\diagnostics\full100_correspondence_audit"
$assetRoot = "work\model_assets_a0"
```

预检输入、标签、证据、模型资产和资源门：

```powershell
& $pyExe "outputs\q1\v1_delivery\code\run_q1_v1.py" --stage prepare --input-root $inputRoot --audit-root $auditRoot --asset-root $assetRoot --output-root "outputs\q1\v1_delivery"
```

完整100条正式生产或对已完成任务做哈希验证后续跑：

```powershell
& $pyExe "outputs\q1\v1_delivery\code\run_q1_v1.py" --stage all --input-root $inputRoot --audit-root $auditRoot --asset-root $assetRoot --output-root "outputs\q1\v1_delivery" --resume
```

`--resume`只复用与冻结的文本、源MP4、配置、模型资产和输出哈希一致的结果。资源低于1GiB时会在样本边界暂停；恢复可用内存后用同一命令续跑。其他阶段可用 `--stage text` 或 `--stage media` 单独运行。不要修改参数后继续复用旧结果。

验证100条正式结果：

```powershell
& $pyExe "outputs\q1\v1_delivery\code\validate_q1_v1.py" --output-root "outputs\q1\v1_delivery" --input-root $inputRoot --audit-root $auditRoot --asset-root $assetRoot --expected-count 100
```

## 独立分支复现检查

README对应的实际复现使用新的空目录 `outputs/q1/v1_delivery/repro_check`，覆盖普通片段级、静音、错配、无脸、零时长诊断、最长文本等分支样本。固定样本来自 `reports/branch_check_samples.json`。重新执行：

```powershell
$sampleKeys = @(
  '-3g5yACwYnA$_$13', '-571d8cVauQ$_$0', '-a55Q6RWvTA$_$3',
  '-mJ2ud6oKI8$_$1', '-mJ2ud6oKI8$_$2', '-mJ2ud6oKI8$_$6',
  '-s9qJ7ATP7w$_$0', '-s9qJ7ATP7w$_$6', '-yRb-Jum7EQ$_$1'
)
$sampleArgs = @()
foreach ($sampleKey in $sampleKeys) { $sampleArgs += ('--sample-key={0}' -f $sampleKey) }
& $pyExe "outputs\q1\v1_delivery\code\run_q1_v1.py" --stage all --input-root $inputRoot --audit-root $auditRoot --asset-root $assetRoot --output-root "outputs\q1\v1_delivery\repro_check" --resume @sampleArgs
& $pyExe "outputs\q1\v1_delivery\code\validate_q1_v1.py" --output-root "outputs\q1\v1_delivery\repro_check" --input-root $inputRoot --audit-root $auditRoot --asset-root $assetRoot --expected-count 9
& $pyExe "outputs\q1\v1_delivery\code\compare_q1_reproduction.py" --reference-root "outputs\q1\v1_delivery" --reproduction-root "outputs\q1\v1_delivery\repro_check"
```

复现报告逐样本逐数组比较离散字段、mask、中心索引和浮点特征；容差为 `rtol=1e-6, atol=1e-6`。该复现范围只覆盖上面固定的分支样本，不宣称完整100条第二次重跑。

## Q1候选附件包

`package/q1_v1_candidate_q1_closed_2026-09-24.zip`为闭合修复后候选包，保存100个正式NPZ、合同、代码、结果表、方法与验收报告、完整对应审计证据、A0保护基准和必要元数据。包内保留运行所需目录结构。原始附件1和大模型权重不重复打包；按上文的原始输入清单、固定模型版本和SHA-256取得，并在运行前完成校验。原始`q1_v1_candidate.zip`基线保持不变。

`package_manifest.json`和`size_report.json`作为与ZIP并列的sidecar文件保存，不打入ZIP；前者逐文件记录候选包内容的字节数及SHA-256，后者记录压缩包实测大小和50,000,000 bytes预算余量。该数值是Q1候选包大小，不代表全题最终附件已完成50 MB核算。

## 统一读取

```powershell
& $pyExe "outputs\q1\v1_delivery\code\read_q1_sample.py" "outputs\q1\v1_delivery\features\-s9qJ7ATP7w___6.npz"
```

接口保留不同模态的原生长度：文本为 `L×768`，音频为 `K_audio×25`，视频为 `K_video×52`；只有具备可信词时间的样本才使用逐词 `L×50` audio 和 `L×104` vision 派生量。运行 `feature_reader.load_q1_feature(path)` 可取得全部数值数组和mask；接口不做补齐，也不把缺失视觉观测改成有效值。

## 解释边界

结构验证PASS只说明文件、来源、时间字段、长度、mask和聚合复算符合合同。它不表示100条官方文本全部与实际人声一致，也不提供forced-alignment准确率或情感分类性能。路由数量以 `reports/results_summary.json` 和100行manifest为准；静音、错配、证据不足和无脸都保留为独立状态。
