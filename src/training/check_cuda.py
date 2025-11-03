import torch
import sys

print("=== PyTorch环境诊断 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"Python版本: {sys.version}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {getattr(torch.version, 'cuda', 'N/A')}")
print(f"GPU设备数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA设备: 未检测到")