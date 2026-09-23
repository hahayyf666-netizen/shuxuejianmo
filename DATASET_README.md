# E题数据

两个大型 PKL 文件已按 90 MiB 做原始字节分片，以适配 GitHub 普通 Git 的单文件限制。
下载仓库后，在仓库根目录运行 `python merge_dataset.py merge --manifest dataset_manifest.json`，脚本会校验 SHA-256 并还原原始文件。
