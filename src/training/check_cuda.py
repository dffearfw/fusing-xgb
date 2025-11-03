import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("CUDA不可用，请检查:")
    print("1. 是否安装了CUDA版本的PyTorch")
    print("2. 显卡驱动是否正常")
    print("3. CUDA工具包是否安装")