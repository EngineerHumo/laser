import torch
import sys

print("=" * 60)
print("PyTorch CUDA 安装验证")
print("=" * 60)
print(f"PyTorch版本: {torch.__version__}")
print(f"Python版本: {sys.version}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 实际GPU计算测试
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"GPU计算测试成功: 矩阵乘法结果形状 {z.shape}")
    print("🎉 PyTorch CUDA安装完全成功！")
else:
    print("❌ 还有问题，CUDA仍不可用")