# SparseDrive Profiling Guide (NVTX + Nsight Systems)

本指南介绍 SparseDrive 的 NVTX 标注结构与使用 Nsight Systems（nsys）进行性能分析的方法。

## 1. NVTX 标注层次

顶层（`tools/train.py`）
- Main Function
- Model Initialization
- Dataset Preparation
- Training Process

训练内层（`projects/mmdet3d_plugin/apis/mmdet_train.py`）
- DataLoader: Build Dataloaders
- Model: To CUDA + Wrap (DP/DDP)
- Runner: Build + Optimizer
- Runner: FP16/Optimizer Config
- Runner: Register Hooks
- Eval: Build Val Dataloader + Hook
- Runner: run() Iteration Loop
- Per-iter：`NVTXIterHook` 在每个 `train_iter` 前后自动 push/pop

说明：
- 内层范围避免与外层重复命名，清晰区分职责与阶段。
- 迭代级 NVTX Hook 不侵入模型前向代码，开销低、效果好。

## 2. 运行 Nsight Systems

已有脚本：`scripts/nsys_profile.sh`

示例：
```bash
# Stage 1：全量采样（CUDA + cuBLAS + cuDNN + NVTX + CPU采样）
STAGE=1 MODE=full CUDA_VISIBLE_DEVICES=0 GPUS=1 bash scripts/nsys_profile.sh

# Stage 2：低开销采样（CUDA + NVTX）
STAGE=2 MODE=light CUDA_VISIBLE_DEVICES=0 GPUS=1 bash scripts/nsys_profile.sh
```

生成：
- 时间线：`nsys_reports/stage{1|2}_{full|light}_<ts>.qdrep`
- 统计：`nsys_reports/stage{1|2}_{full|light}_<ts>_stats.html`

## 3. 分析建议

- 先看顶层区间占比，定位瓶颈（数据、模型、优化器、主循环）。
- 展开 `train_iter N`，观察单步耗时、Kernel 密度与重叠。
- 打开 cuBLAS/cuDNN 视图，关注 GEMM/卷积热点。
- 如需更细粒度，再在关键模块临时加 NVTX，但以 `NVTXIterHook` 为主，避免噪声。

## 4. 注意事项

- `MODE=light` 低开销适合长跑；`MODE=full` 适合短窗口深度分析。
- 无 GPU 时 `tools/train.py` 的 NVTX 自动降级为空实现。
- 建议配合固定随机种子与 `--deterministic` 做对比实验。


