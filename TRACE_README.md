# SparseDrive Python函数Trace配置

本项目为SparseDrive自动驾驶系统提供了完整的Python函数trace配置，使用NVIDIA Nsight Systems进行性能分析和优化。

## 📁 文件结构

```
.
├── sparsedrive_trace.json          # 主要的trace配置文件
├── run_trace.sh                    # 快速启动脚本
├── test_trace.py                   # 测试脚本
├── docs/
│   ├── quick_start.md              # 快速开始指南
│   ├── sparsedrive_trace_guide.md # 详细trace指南
│   └── stage1_vs_stage2.md        # Stage1 vs Stage2对比
└── TRACE_README.md                 # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装NVIDIA Nsight Systems
sudo dpkg -i NsightSystems-linux-cli-public-*.deb

# 检查安装
nsys --version
```

### 2. 运行测试

```bash
# 运行测试脚本验证配置
./run_trace.sh -t

# 或者直接运行
python test_trace.py
```

### 3. 开始trace分析

```bash
# 使用快速启动脚本
./run_trace.sh your_sparsedrive_script.py

# 或者手动运行
nsys profile --trace=cuda,nvtx,osrt,python \
    --python-trace-backend=py-spy \
    --python-trace-config=sparsedrive_trace.json \
    --output=sparsedrive_trace \
    python your_script.py
```

## 📊 Trace配置详解

### 训练阶段概述

SparseDrive采用两阶段训练策略：

#### 🎯 Stage1: 检测 + 建图阶段
- **主要任务**: 3D目标检测 + 高清地图重建
- **关键模块**: 特征提取、3D检测头、地图解码器、注意力机制
- **性能重点**: 特征提取效率、检测精度、建图质量

#### 🚗 Stage2: 运动规划阶段  
- **主要任务**: 在Stage1基础上增加运动预测和轨迹规划
- **关键模块**: 运动规划头、运动解码器、实例队列、时序实例管理
- **性能重点**: 时序建模效率、运动预测精度、规划实时性

**注意**: Stage2必须在Stage1完成后进行，需要加载Stage1的预训练权重。

### 主要模块覆盖

| 阶段 | 模块 | 颜色 | 关键函数 | 说明 |
|------|------|------|----------|------|
| **Stage1** | SparseDrive核心 | 🟢 | `extract_feat`, `forward` | 主模型架构 |
| **Stage1** | 检测头 | 🔵 | `forward`, `loss`, `post_process` | 3D检测和跟踪 |
| **Stage1** | 地图模块 | 🟡 | `decode_map` | 在线建图 |
| **Stage1** | 注意力机制 | ⚫ | `MultiHeadAttention` | 稀疏注意力计算 |
| **Stage2** | 运动规划 | 🟣 | `predict_motion`, `plan_trajectory` | 轨迹预测和规划 |
| **Stage2** | 实例管理 | 🔵 | `update`, `query` | 时序建模 |
| **Stage2** | 运动解码 | 🟠 | `decode_trajectory` | 运动轨迹解码 |

### 性能分析重点

1. **特征提取阶段**
   - 多视角图像处理性能
   - GridMask和深度分支耗时
   - 特征图内存使用

2. **稀疏感知阶段**
   - 3D检测头推理时间
   - 注意力机制计算复杂度
   - 稀疏操作效率

3. **运动规划阶段**
   - 轨迹预测和规划时间
   - 实例队列更新频率
   - 碰撞检测性能

4. **内存管理**
   - GPU内存峰值使用
   - 特征图存储优化
   - 批处理大小调优

## 🛠️ 使用方法

### 基本命令

```bash
# 查看帮助
./run_trace.sh -h

# 显示配置
./run_trace.sh -c

# 运行测试
./run_trace.sh -t

# 运行Stage1训练trace
./run_trace.sh -s1

# 运行Stage2训练trace
./run_trace.sh -s2

# 查看报告
./run_trace.sh -v

# 运行自定义脚本
./run_trace.sh your_script.py arg1 arg2
```

### 环境变量配置

```bash
export NSYS_PROFILE=1
export NSYS_PROFILE_PYTHON_TRACE_CONFIG=sparsedrive_trace.json
python your_script.py
```

### 代码集成

```python
import torch.cuda.nvtx as nvtx

class SparseDrive(nn.Module):
    def forward(self, img, **data):
        nvtx.range_push("SparseDrive.forward")
        try:
            # 你的代码
            return result
        finally:
            nvtx.range_pop()
```

## 📈 分析结果

### 生成的文件

- `*.nsys-rep`: Nsight Systems报告文件
- `*.sqlite`: SQLite格式的trace数据
- `*.qdrep`: 可选的数据收集文件

### 查看报告

```bash
# 使用GUI查看
nsys-ui sparsedrive_trace_*.nsys-rep

# 导出为SQLite
nsys export --type sqlite --output trace.sqlite *.nsys-rep

# 命令行分析
nsys stats *.nsys-rep
```

## 🔧 优化建议

### 模型优化

- 使用混合精度训练 (FP16)
- 启用CUDA图优化
- 优化稀疏卷积操作
- 使用梯度检查点

### 数据优化

- 多进程数据加载
- 预加载和缓存数据
- 优化数据增强管道
- 动态批处理大小

### 内存优化

- 监控GPU内存使用
- 优化特征图存储
- 使用内存池
- 梯度累积

## 🐛 常见问题

### 1. nsys命令未找到

```bash
# 检查安装
which nsys
# 重新安装
sudo dpkg -i NsightSystems-*.deb
```

### 2. Python trace失败

```bash
# 检查py-spy后端
pip install py-spy
# 或者使用其他后端
--python-trace-backend=py-spy
```

### 3. 权限问题

```bash
# 确保有足够权限
sudo chmod +x run_trace.sh
# 检查文件权限
ls -la *.sh *.py
```

## 📚 参考资料

- [NVIDIA Nsight Systems文档](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Profiler](https://pytorch.org/tutorials/profiler.html)
- [SparseDrive论文](https://arxiv.org/abs/2405.19620)
- [MMDetection3D文档](https://mmdetection3d.readthedocs.io/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进trace配置：

1. Fork本项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目遵循与SparseDrive相同的许可证。

---

**注意**: 使用trace功能可能会对性能产生轻微影响，建议在开发/调试环境中使用，生产环境中谨慎使用。
