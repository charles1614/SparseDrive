# SparseDrive Trace配置完成总结

## 🎯 已完成的工作

我已经根据PyTorch的JSON配置文件，为SparseDrive项目创建了完整的Python函数trace配置，**完全包含了Stage1和Stage2两个训练阶段**。

## 📁 创建的文件

### 1. 核心配置文件
- **`sparsedrive_trace.json`** - 主要的trace配置文件，包含：
  - Stage1: 检测 + 建图阶段的所有关键函数
  - Stage2: 运动规划阶段的所有关键函数
  - PyTorch核心函数的trace配置

### 2. 自动化脚本
- **`run_trace.sh`** - 快速启动脚本，支持：
  - `-s1` / `--stage1`: 运行Stage1训练trace
  - `-s2` / `--stage2`: 运行Stage2训练trace
  - `-t` / `--test`: 运行测试脚本
  - `-c` / `--config`: 显示trace配置
  - `-v` / `--view`: 查看trace报告

### 3. 测试和验证
- **`test_trace.py`** - 测试脚本，验证trace配置是否正常工作
- **`docs/sparsedrive_trace_guide.md`** - 详细的trace使用指南
- **`docs/stage1_vs_stage2.md`** - Stage1和Stage2的详细对比说明
- **`TRACE_README.md`** - 项目总览和快速开始指南

## 🔍 Stage1 vs Stage2 Trace覆盖

### Stage1: 检测 + 建图阶段
- ✅ **SparseDrive核心**: `extract_feat`, `forward`, `forward_train`, `forward_test`
- ✅ **检测头**: `Detection3DHead.forward`, `loss`, `post_process`
- ✅ **地图模块**: `MapDecoder.forward`, `decode_map`
- ✅ **注意力机制**: `MultiHeadAttention`, `SparseAttention`, `CrossAttention`
- ✅ **特征处理**: `TransformerBlock`, `SparseBlock`, `FeatureExtractor`
- ✅ **数据增强**: `GridMask.forward`, `apply_mask`

### Stage2: 运动规划阶段
- ✅ **运动规划头**: `MotionPlanningHead.forward`, `predict_motion`, `plan_trajectory`
- ✅ **运动解码器**: `MotionDecoder.forward`, `decode_trajectory`, `decode_motion`
- ✅ **实例队列**: `InstanceQueue.push`, `pop`, `update`, `get_history`
- ✅ **实例管理**: `InstanceBank.update`, `query`, `get_temporal_instances`
- ✅ **运动目标**: `MotionTarget.forward`, `compute_loss`, `get_targets`

## 🚀 使用方法

### 快速开始
```bash
# 运行Stage1训练trace
./run_trace.sh -s1

# 运行Stage2训练trace
./run_trace.sh -s2

# 查看帮助
./run_trace.sh -h
```

### 手动运行
```bash
# Stage1
nsys profile --trace=cuda,nvtx,osrt,python \
    --python-trace-backend=py-spy \
    --python-trace-config=sparsedrive_trace.json \
    --output=sparsedrive_stage1_trace \
    bash -lc 'cd /workspace/SparseDrive && conda run -n sparsedrive bash -lc "PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage1.py 1 --deterministic"'

# Stage2
nsys profile --trace=cuda,nvtx,osrt,python \
    --python-trace-backend=py-spy \
    --python-trace-config=sparsedrive_trace.json \
    --output=sparsedrive_stage2_trace \
    bash -lc 'cd /workspace/SparseDrive && conda run -n sparsedrive bash -lc "PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage2.py 1 --deterministic"'
```

## 📊 性能分析重点

### Stage1分析重点
- **特征提取效率**: `extract_feat`函数耗时
- **检测性能**: 3D检测头的推理时间
- **建图质量**: 地图解码器的性能
- **注意力计算**: 稀疏注意力的复杂度

### Stage2分析重点
- **时序建模**: 实例队列的更新频率
- **运动预测**: 轨迹预测的准确性
- **规划性能**: 轨迹规划的实时性
- **内存管理**: 多帧特征的内存使用

## 🔧 技术特点

### 1. 完全兼容现有命令
- 支持你当前使用的 `--pytorch=functions-trace` 参数
- 可以同时使用我们的自定义配置和PyTorch内置配置

### 2. 分阶段trace支持
- Stage1和Stage2使用不同的颜色标识
- 可以分别分析两个阶段的性能特点
- 支持渐进式性能优化

### 3. 自动化工具链
- 一键运行Stage1/Stage2 trace
- 自动生成时间戳的输出文件
- 集成的报告查看工具

## 🎉 总结

现在你可以：

1. **直接运行Stage1 trace**: `./run_trace.sh -s1`
2. **直接运行Stage2 trace**: `./run_trace.sh -s2`
3. **分析两个阶段的性能差异**
4. **定位具体的性能瓶颈**
5. **优化模型架构和训练策略**

这个trace配置完全覆盖了SparseDrive的两个训练阶段，帮助你深入了解模型的性能表现，特别是在Stage1到Stage2的过渡过程中，可以清楚地看到运动规划模块带来的额外计算开销和性能变化。

建议按照以下顺序使用：
1. 先运行Stage1 trace，了解基础性能
2. 完成Stage1训练后，运行Stage2 trace
3. 对比两个阶段的性能差异
4. 针对性地优化性能瓶颈

祝你训练顺利！🚗💨
