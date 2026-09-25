# Stage C-4 复现入口

本目录仅重建附件4 `01–20` 的候选特征并与官方 PKL 比较。不会加载 Q3 预测器，也不会训练或推理。已有结果和逐样本证据见 `Q3_FEATURE_RECONSTRUCTION_REPORT.md`、`results/`；`FEATURE_SPACE_ATTRIBUTION_SCOPE_FREEZE.md` 是技术备选边界，现行 Q3 协议与训练 STOP 门未改。

## 输入与固定资产

- 从本仓库根目录运行下列 PowerShell 命令。`$dataRoot` 应指向原样附件4目录；脚本会按 C-2 清单重新核对未对齐 PKL/MP4 的 SHA-256。
- Python 环境：本次为 Python 3.12.10；包版本见 `results/c4_gate.json`。
- BERT tokenizer：本次使用 `work/q3_mapping_probe/vocab.txt`，SHA-256 `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`。
- BERT 模型：从 [google-bert/bert-base-uncased 固定版本](https://huggingface.co/google-bert/bert-base-uncased/tree/8229d58a8e9c4f761cdb4a3f0434f856e1ae1d5d) 获取 `config.json` 和 `model.safetensors`；后者 SHA-256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。该大文件未纳入仓库。`$modelDir` 应是包含二者的目录。
- MediaPipe face task：本次使用本地 `work/model_assets_a0/mediapipe/face_landmarker.task`；音视频只做未证实的候选重建，不能把它当作官方提取器。

## 依次执行

```powershell
$repo = (Get-Location).Path
$stage = Join-Path $repo '阶段4.2_Q3_v1_preflight/stage_c4_feature_reconstruction'
$results = Join-Path $stage 'results'
$dataRoot = 'D:\Workspace\数学建模\E题数据\附件4-可解释专项视频样本与特征文件'
$python = 'C:\Users\Fine\Documents\Codex\2026-09-23\https-github-com-hahayyf666-netizen-shuxuejianmo\work\.venv_q1\Scripts\python.exe'
$vocab = 'C:\Users\Fine\Documents\Codex\2026-09-23\https-github-com-hahayyf666-netizen-shuxuejianmo\work\q3_mapping_probe\vocab.txt'
$modelDir = 'C:\Users\Fine\Documents\Codex\2026-09-23\https-github-com-hahayyf666-netizen-shuxuejianmo\work\q3_reconstruction_assets\hf_cache\models--google-bert--bert-base-uncased\snapshots\8229d58a8e9c4f761cdb4a3f0434f856e1ae1d5d'
$faceTask = 'C:\Users\Fine\Documents\Codex\2026-09-23\https-github-com-hahayyf666-netizen-shuxuejianmo\work\model_assets_a0\mediapipe\face_landmarker.task'
$inventory = Join-Path $repo '阶段4.2_Q3_v1_preflight/stage_c2_provenance/results/pair_inventory_20.csv'
$core = Join-Path $repo '阶段2.1.7_A0_Pilot/Q1第一版特征生产_2026-09-24/code'

Get-FileHash -LiteralPath (Join-Path $modelDir 'model.safetensors') -Algorithm SHA256
Get-FileHash -LiteralPath $vocab -Algorithm SHA256
& $python (Join-Path $stage 'reconstruct_text.py') --aligned (Join-Path $dataRoot '对齐版本') --vocab $vocab --model-dir $modelDir --out $results
& $python (Join-Path $stage 'reconstruct_media.py') --unaligned (Join-Path $dataRoot '未对齐版本') --pair-inventory $inventory --q1-core-dir $core --face-task $faceTask --out $results
& $python (Join-Path $stage 'analyze_similarity_controls.py') --unaligned (Join-Path $dataRoot '未对齐版本') --candidate-dir (Join-Path $results 'candidate_npz') --out $results
& $python (Join-Path $stage 'summarize_c4.py') --results $results --pair-inventory $inventory
```

脚本按顺序执行，任何非零退出码都应中断复现并记录。文本结果以 `≤1e-4` 逐维容差判定，不声称 bitwise 相等；音频和视觉的 CKA 只是一项代理诊断，不计入严格行级恢复率。
