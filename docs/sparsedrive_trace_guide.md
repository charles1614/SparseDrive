# SparseDrive Python函数Trace指南

本指南介绍如何使用NVIDIA Nsight Systems来trace SparseDrive项目的Python函数，帮助分析模型性能和定位性能瓶颈。

## SparseDrive训练阶段概述

SparseDrive采用两阶段训练策略，每个阶段有不同的任务和性能特点：

### Stage1: 检测 + 建图阶段
- **主要任务**: 3D目标检测 + 高清地图重建
- **关键模块**: 
  - 特征提取 (`extract_feat`)
  - 3D检测头 (`Detection3DHead`)
  - 地图解码器 (`MapDecoder`)
  - 注意力机制 (`MultiHeadAttention`, `SparseAttention`)
- **性能重点**: 特征提取效率、检测精度、建图质量

### Stage2: 运动规划阶段  
- **主要任务**: 在Stage1基础上增加运动预测和轨迹规划
- **关键模块**:
  - 运动规划头 (`MotionPlanningHead`)
  - 运动解码器 (`MotionDecoder`)
  - 实例队列 (`InstanceQueue`)
  - 时序实例管理 (`InstanceBank`)
- **性能重点**: 时序建模效率、运动预测精度、规划实时性

**注意**: Stage2必须在Stage1完成后进行，需要加载Stage1的预训练权重。

## 概述

SparseDrive是一个基于稀疏场景表示的端到端自动驾驶系统，包含以下主要组件：
- **稀疏感知模块**：3D检测、跟踪和在线建图
- **运动预测模块**：轨迹预测和规划
- **实例记忆队列**：时序建模

## Trace配置文件

我们创建了 `sparsedrive_trace.json` 配置文件，包含以下主要域：

### 1. SparseDrive核心模块 (绿色 - 0x4CAF50)
- `SparseDrive.extract_feat`: 特征提取
- `SparseDrive.forward`: 前向传播
- `SparseDrive.forward_train`: 训练模式
- `SparseDrive.forward_test`: 测试模式

### 2. 检测头模块 (蓝色 - 0x2196F3)
- `SparseDriveHead.forward`: 检测头前向传播
- `SparseDriveHead.loss`: 损失计算
- `SparseDriveHead.post_process`: 后处理

### 3. 3D检测模块 (橙色 - 0xFF9800)
- `Detection3DHead.forward`: 3D检测前向传播
- `Detection3DHead.loss`: 检测损失
- `Detection3DHead.post_process`: 检测后处理

### 4. 运动规划模块 (紫色 - 0x9C27B0)
- `MotionPlanningHead.forward`: 运动规划前向传播
- `MotionPlanningHead.predict_motion`: 运动预测
- `MotionPlanningHead.plan_trajectory`: 轨迹规划

### 5. 地图模块 (粉色 - 0xE91E63)
- `MapDecoder.forward`: 地图解码
- `MapDecoder.decode_map`: 地图解码

### 6. 注意力机制 (深灰 - 0x607D8B)
- `MultiHeadAttention.forward`: 多头注意力
- `SparseAttention.forward`: 稀疏注意力
- `CrossAttention.forward`: 交叉注意力

### 7. 核心块模块 (棕色 - 0x795548)
- `TransformerBlock.forward`: Transformer块
- `SparseBlock.forward`: 稀疏块
- `FeatureExtractor.forward`: 特征提取器

### 8. 实例管理 (青色 - 0x00BCD4)
- `InstanceBank.update`: 实例库更新
- `InstanceBank.query`: 实例查询
- `InstanceBank.add_instance`: 添加实例

### 9. 运动实例队列 (浅绿 - 0x8BC34A)
- `InstanceQueue.push`: 推入队列
- `InstanceQueue.pop`: 弹出队列
- `InstanceQueue.update`: 更新队列

### 10. 运动解码器 (深橙 - 0xFF5722)
- `MotionDecoder.forward`: 运动解码器前向传播
- `MotionDecoder.decode_trajectory`: 轨迹解码

### 11. 地图块模块 (靛蓝 - 0x3F51B5)
- `MapBlock.forward`: 地图块前向传播
- `MapFeatureExtractor.forward`: 地图特征提取

### 12. 网格掩码 (青绿 - 0x009688)
- `GridMask.forward`: 网格掩码前向传播
- `GridMask.apply_mask`: 应用掩码

## 使用方法

### 1. 安装Nsight Systems

```bash
# 下载并安装Nsight Systems CLI
wget https://developer.nvidia.com/nsight-systems/get-started
# 或者使用已下载的deb包
sudo dpkg -i NsightSystems-linux-cli-public-2025.4.1.172-3634357.deb
```

### 2. 运行Python程序时启用trace

```bash
# 使用nsys profile命令运行Python程序
nsys profile --trace=cuda,nvtx,osrt,python --python-trace-backend=py-spy \
    --python-trace-config=sparsedrive_trace.json \
    python your_sparsedrive_script.py

# 或者使用环境变量
export NSYS_PROFILE=1
export NSYS_PROFILE_PYTHON_TRACE_CONFIG=sparsedrive_trace.json
python your_sparsedrive_script.py
```

### 3. 在代码中添加NVTX标记

```python
import torch.cuda.nvtx as nvtx

class SparseDrive(nn.Module):
    def forward(self, img, **data):
        nvtx.range_push("SparseDrive.forward")
        try:
            if self.training:
                result = self.forward_train(img, **data)
            else:
                result = self.forward_test(img, **data)
            return result
        finally:
            nvtx.range_pop()
```

### 4. 分析trace结果

```bash
# 生成HTML报告
nsys export --type sqlite --output sparsedrive_trace.sqlite nsys-report-*.nsys-rep

# 使用Nsight Systems GUI查看
nsys-ui nsys-report-*.nsys-rep
```

## 性能分析重点

### Stage1: 检测 + 建图阶段
- **特征提取**: 关注 `extract_feat` 函数的执行时间
- **多视角处理**: 分析多视角图像处理性能
- **GridMask**: 检查GridMask和深度分支的性能
- **3D检测**: 监控3D检测头的推理时间
- **建图**: 分析地图解码器的性能
- **注意力机制**: 检查稀疏注意力的耗时

### Stage2: 运动规划阶段
- **运动预测**: 关注轨迹预测和规划的时间
- **实例管理**: 分析实例队列的更新频率
- **时序建模**: 检查时序特征的处理效率
- **碰撞检测**: 分析碰撞检测的性能
- **轨迹生成**: 监控运动解码器的性能
- **目标计算**: 分析运动目标的计算开销

### 通用性能指标
- **内存使用**: 监控GPU内存占用和峰值
- **计算效率**: 分析各模块的FLOPs和吞吐量
- **数据流**: 检查特征图的内存传输效率

### 4. 内存使用
- 监控实例库的内存占用
- 分析特征图的内存使用
- 检查GPU内存的峰值使用

## 常见性能问题

### 1. 数据加载瓶颈
- 检查DataLoader的性能
- 分析数据预处理时间
- 优化数据增强操作

### 2. 模型推理瓶颈
- 分析各模块的执行时间
- 检查注意力计算的复杂度
- 优化稀疏操作

### 3. 内存瓶颈
- 监控GPU内存使用
- 分析特征图的存储
- 优化批处理大小

## 优化建议

### 1. 模型优化
- 使用混合精度训练 (FP16)
- 启用CUDA图优化
- 优化稀疏卷积操作

### 2. 数据优化
- 使用多进程数据加载
- 预加载和缓存数据
- 优化数据增强管道

### 3. 内存优化
- 使用梯度检查点
- 优化特征图存储
- 动态批处理大小

## 示例脚本

创建一个简单的trace测试脚本：

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from projects.mmdet3d_plugin.models.sparsedrive import SparseDrive

def test_sparsedrive_trace():
    # 创建模型
    model = SparseDrive(
        img_backbone=dict(type='ResNet'),
        head=dict(type='SparseDriveHead')
    )
    
    # 创建测试数据
    batch_size = 2
    num_cams = 6
    img = torch.randn(batch_size, num_cams, 3, 256, 704)
    
    # 启用trace
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        # 前向传播
        output = model(img)
        
        # 反向传播
        if model.training:
            loss = output['loss']
            loss.backward()
    
    # 打印结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    test_sparsedrive_trace()
```

## 总结

通过使用这个trace配置文件，你可以：
1. 深入了解SparseDrive各模块的性能表现
2. 定位性能瓶颈和优化点
3. 监控训练和推理过程中的关键指标
4. 优化模型架构和训练策略

建议定期运行trace分析，特别是在模型架构变更或超参数调整后，以确保性能的持续改进。
