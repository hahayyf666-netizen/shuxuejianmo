# Verified environment

Environment checked for this candidate's local inference and figures:

- OS: Windows 11, 64-bit
- Python: 3.12.10, 64-bit
- PyTorch: 2.8.0+cpu
- NumPy: 2.5.0
- Matplotlib: 3.11.0
- CUDA: unavailable locally; candidate inference uses CPU
- GPU: none used locally

The DR-X training environment used two NVIDIA GeForce RTX 3090 GPUs with CUDA 12.8-enabled PyTorch 2.8.0. The frozen training summary records the delivery run on `cuda:1`. Local CPU inference is used only for validation/error analysis and frozen-output reproduction. Small floating-point differences can occur across hardware or library kernels; comparisons use the frozen absolute tolerance `1e-6`.

This records the runtime actually checked; it is not a complete `pip freeze`.
