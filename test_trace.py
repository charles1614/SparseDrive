#!/usr/bin/env python3
"""
SparseDrive Trace测试脚本
用于验证trace配置是否正常工作
"""

import os
import sys
import torch
import torch.nn as nn
import time

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'projects'))

def create_dummy_model():
    """创建一个简化的SparseDrive模型用于测试"""
    
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return [x]
    
    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(64, 128, 3, padding=1)
            # 动态计算输入特征维度
            self.linear = None
            
        def _init_linear(self, input_features, device):
            """动态初始化线性层"""
            if self.linear is None:
                self.linear = nn.Linear(input_features, 100).to(device)
            
        def forward(self, features, data=None, **kwargs):
            x = self.conv(features[0])
            x = x.flatten(1)
            # 动态初始化线性层
            self._init_linear(x.shape[1], x.device)
            x = self.linear(x)
            return {'output': x}
            
        def loss(self, outputs, data):
            return {'loss': torch.mean(outputs['output'])}
            
        def post_process(self, outputs, data):
            return [{'result': outputs['output']}]
    
    class DummySparseDrive(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()
            self.head = DummyHead()
            
        def extract_feat(self, img):
            """特征提取"""
            if img.dim() == 5:  # multi-view
                bs, num_cams = img.shape[:2]
                img = img.flatten(end_dim=1)
            else:
                bs, num_cams = img.shape[0], 1
                
            features = self.backbone(img)
            
            # 重塑特征 - 修复维度问题
            for i, feat in enumerate(features):
                if feat.dim() == 4:  # [bs*num_cams, channels, height, width]
                    features[i] = feat.view(bs, num_cams, feat.shape[1], feat.shape[2], feat.shape[3])
                else:
                    features[i] = feat.view(bs, num_cams, -1)
            return features
            
        def forward(self, img, **data):
            """前向传播"""
            if self.training:
                return self.forward_train(img, **data)
            else:
                return self.forward_test(img, **data)
                
        def forward_train(self, img, **data):
            """训练模式"""
            features = self.extract_feat(img)
            # 将多视角特征展平为单视角特征
            flattened_features = []
            for feat in features:
                if feat.dim() == 5:  # [bs, num_cams, channels, height, width]
                    bs, num_cams = feat.shape[:2]
                    flattened_features.append(feat.flatten(end_dim=1))
                else:
                    flattened_features.append(feat)
            outputs = self.head(flattened_features, data)
            loss = self.head.loss(outputs, data)
            return loss
            
        def forward_test(self, img, **data):
            """测试模式"""
            features = self.extract_feat(img)
            # 将多视角特征展平为单视角特征
            flattened_features = []
            for feat in features:
                if feat.dim() == 5:  # [bs, num_cams, channels, height, width]
                    bs, num_cams = feat.shape[:2]
                    flattened_features.append(feat.flatten(end_dim=1))
                else:
                    flattened_features.append(feat)
            outputs = self.head(flattened_features, data)
            results = self.head.post_process(outputs, data)
            return results
    
    return DummySparseDrive()

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    model = create_dummy_model()
    model.train()
    
    # 创建测试数据
    batch_size = 2
    num_cams = 6
    img = torch.randn(batch_size, num_cams, 3, 256, 704)
    
    # 测试前向传播
    try:
        output = model(img)
        print(f"✓ 前向传播成功，输出形状: {output['loss'].shape}")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 测试反向传播
    try:
        output['loss'].backward()
        print("✓ 反向传播成功")
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        return False
    
    return True

def test_trace_integration():
    """测试trace集成"""
    print("\n测试trace集成...")
    
    # 检查是否可以导入trace相关模块
    try:
        import torch.cuda.nvtx as nvtx
        print("✓ NVTX模块可用")
    except ImportError:
        print("✗ NVTX模块不可用")
        return False
    
    # 检查trace配置文件
    trace_config = "sparsedrive_trace.json"
    if os.path.exists(trace_config):
        print(f"✓ Trace配置文件存在: {trace_config}")
    else:
        print(f"✗ Trace配置文件不存在: {trace_config}")
        return False
    
    return True

def test_performance_monitoring():
    """测试性能监控"""
    print("\n测试性能监控...")
    
    model = create_dummy_model()
    model.eval()
    
    # 预热
    batch_size = 1
    num_cams = 6
    img = torch.randn(batch_size, num_cams, 3, 256, 704)
    
    with torch.no_grad():
        for _ in range(3):
            _ = model(img)
    
    # 性能测试
    num_iterations = 10
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(img)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    
    print(f"✓ 平均推理时间: {avg_time*1000:.2f} ms")
    print(f"✓ 推理FPS: {fps:.2f}")
    
    return True

def main():
    """主函数"""
    print("SparseDrive Trace测试")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
    
    print("\n" + "=" * 50)
    
    # 运行测试
    tests = [
        test_basic_functionality,
        test_trace_integration,
        test_performance_monitoring
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以开始使用trace功能。")
        print("\n下一步:")
        print("1. 使用 nsys profile 命令运行你的SparseDrive脚本")
        print("2. 查看生成的trace报告")
        print("3. 分析性能瓶颈")
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
