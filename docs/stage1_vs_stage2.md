# SparseDrive Stage1 vs Stage2 详细对比

## 概述

SparseDrive采用两阶段训练策略，每个阶段专注于不同的任务，通过分阶段训练提高模型性能和训练稳定性。

## 🎯 Stage1: 检测 + 建图阶段

### 主要任务
- **3D目标检测**: 检测车辆、行人、交通标志等3D目标
- **高清地图重建**: 重建道路边界、车道线、交通标志等地图元素
- **特征学习**: 学习多视角图像的特征表示

### 配置文件
```python
# projects/configs/sparsedrive_small_stage1.py
task_config = dict(
    with_det=True,        # 启用检测
    with_map=True,        # 启用建图
    with_motion_plan=False, # 禁用运动规划
)
```

### 关键模块
| 模块 | 功能 | 关键函数 |
|------|------|----------|
| **SparseDrive** | 主模型架构 | `extract_feat`, `forward` |
| **Detection3DHead** | 3D检测头 | `forward`, `loss`, `post_process` |
| **MapDecoder** | 地图解码器 | `forward`, `decode_map` |
| **MultiHeadAttention** | 注意力机制 | `forward` |
| **SparseBlock** | 稀疏特征处理 | `forward` |

### 性能特点
- **计算复杂度**: 中等，主要处理静态场景
- **内存使用**: 相对较低，无时序数据
- **训练稳定性**: 高，任务相对简单
- **收敛速度**: 快，通常20-50个epoch

### 使用场景
- 初始模型训练
- 检测和建图性能调优
- 特征提取器预训练

## 🚗 Stage2: 运动规划阶段

### 主要任务
- **运动预测**: 预测其他车辆和行人的未来轨迹
- **轨迹规划**: 为自车规划安全、高效的行驶轨迹
- **时序建模**: 处理多帧时序数据，建立运动模型

### 配置文件
```python
# projects/configs/sparsedrive_small_stage2.py
task_config = dict(
    with_det=True,        # 启用检测
    with_map=True,        # 启用建图
    with_motion_plan=True, # 启用运动规划
)
```

### 关键模块
| 模块 | 功能 | 关键函数 |
|------|------|----------|
| **MotionPlanningHead** | 运动规划头 | `forward`, `predict_motion`, `plan_trajectory` |
| **MotionDecoder** | 运动解码器 | `forward`, `decode_trajectory` |
| **InstanceQueue** | 实例队列 | `push`, `pop`, `update` |
| **InstanceBank** | 实例库 | `update`, `query`, `get_temporal_instances` |
| **MotionTarget** | 运动目标 | `compute_loss`, `get_targets` |

### 性能特点
- **计算复杂度**: 高，需要处理时序数据和运动预测
- **内存使用**: 高，需要存储多帧特征和实例信息
- **训练稳定性**: 中等，任务复杂度高
- **收敛速度**: 慢，通常需要更多epoch

### 使用场景
- 在Stage1基础上继续训练
- 运动预测和规划性能调优
- 端到端自动驾驶系统训练

## 🔄 训练流程

### 推荐训练顺序

1. **Stage1训练**
   ```bash
   # 训练检测和建图
   ./run_trace.sh -s1
   ```

2. **保存Stage1权重**
   ```bash
   # 在work_dir中找到最新的checkpoint
   ls work_dirs/sparsedrive_small_stage1/
   ```

3. **Stage2训练**
   ```bash
   # 修改配置文件，加载Stage1权重
   # 然后训练运动规划
   ./run_trace.sh -s2
   ```

### 权重加载配置

在Stage2配置文件中添加：
```python
load_from = 'work_dirs/sparsedrive_small_stage1/latest.pth'
```

## 📊 性能对比

### 计算资源需求

| 指标 | Stage1 | Stage2 |
|------|--------|--------|
| **GPU内存** | 8-12GB | 12-16GB |
| **训练时间** | 1-2天 | 2-4天 |
| **收敛epoch** | 20-50 | 50-100 |
| **批处理大小** | 8 | 4 |

### 精度指标

| 任务 | Stage1 | Stage2 |
|------|--------|--------|
| **3D检测mAP** | 0.35-0.45 | 0.35-0.45 |
| **地图IoU** | 0.60-0.70 | 0.60-0.70 |
| **运动预测ADE** | N/A | 0.80-1.20 |
| **轨迹规划ADE** | N/A | 0.60-1.00 |

## 🛠️ Trace分析重点

### Stage1 Trace重点
- **特征提取效率**: `extract_feat`函数耗时
- **检测头性能**: `Detection3DHead.forward`执行时间
- **地图解码**: `MapDecoder.decode_map`性能
- **注意力计算**: 稀疏注意力的计算复杂度

### Stage2 Trace重点
- **时序建模**: 实例队列的更新频率
- **运动预测**: `MotionPlanningHead.predict_motion`耗时
- **轨迹规划**: `MotionPlanningHead.plan_trajectory`性能
- **内存管理**: 多帧特征的内存使用

## 🚨 注意事项

### Stage1注意事项
- 确保数据集中包含足够的3D标注
- 地图标注质量直接影响建图性能
- 多视角图像对齐很重要

### Stage2注意事项
- **必须**先完成Stage1训练
- 时序数据的质量和连续性很关键
- 运动标注的准确性影响很大
- 需要更多的训练数据和计算资源

### 常见问题
1. **Stage2训练不收敛**: 检查Stage1权重是否正确加载
2. **内存不足**: 减少批处理大小或使用梯度检查点
3. **精度下降**: 检查数据质量和标注一致性

## 📈 优化建议

### Stage1优化
- 使用预训练的骨干网络
- 优化数据增强策略
- 调整学习率调度

### Stage2优化
- 使用Stage1的预训练特征
- 优化时序数据采样策略
- 调整运动预测的时间窗口
- 使用更复杂的损失函数

## 🔍 调试技巧

### 使用Trace工具
```bash
# Stage1调试
./run_trace.sh -s1

# Stage2调试  
./run_trace.sh -s2

# 分析特定模块
./run_trace.sh -c  # 查看trace配置
```

### 关键检查点
- **Stage1**: 检测mAP > 0.35, 地图IoU > 0.60
- **Stage2**: 运动预测ADE < 1.20, 轨迹规划ADE < 1.00

通过这种分阶段训练策略，SparseDrive能够在保持检测和建图性能的同时，逐步学习复杂的运动规划能力，最终实现端到端的自动驾驶感知和决策系统。
