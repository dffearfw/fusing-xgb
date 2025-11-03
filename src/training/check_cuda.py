import torch
import subprocess
import os


def comprehensive_cuda_check():
    print("=== 完整CUDA环境检查 ===")

    # 检查环境变量
    cuda_path = os.environ.get('CUDA_PATH', '未设置')
    print(f"CUDA_PATH: {cuda_path}")

    # 检查PATH中的CUDA相关路径
    path = os.environ.get('PATH', '')
    cuda_in_path = any('cuda' in p.lower() for p in path.split(';'))
    print(f"CUDAtoolkit在PATH中: {cuda_in_path}")

    # PyTorch CUDA检查
    print(f"\n=== PyTorch CUDA状态 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {prop.name}")
            print(f"  计算能力: {prop.major}.{prop.minor}")
            print(f"  显存: {prop.total_memory / 1024 ** 3:.1f} GB")

        # CUDA计算测试
        print("\n=== CUDA性能测试 ===")
        device = torch.device('cuda')
        x = torch.randn(5000, 5000, device=device)
        y = torch.randn(5000, 5000, device=device)

        import time
        start = time.time()
        z = x @ y  # 矩阵乘法
        torch.cuda.synchronize()
        end = time.time()

        print(f"5000x5000矩阵乘法耗时: {(end - start) * 1000:.2f} ms")

    else:
        print("\n=== 故障排除建议 ===")
        print("1. 确认已安装CUDA工具包")
        print("2. 检查环境变量CUDA_PATH是否正确设置")
        print("3. 重启命令提示符或IDE")
        print("4. 重新安装支持CUDA的PyTorch版本")


if __name__ == "__main__":
    comprehensive_cuda_check()