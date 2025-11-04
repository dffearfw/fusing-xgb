import os
import sys
import ctypes
import glob


def check_dll_dependencies():
    """检查c10.dll的依赖关系"""
    print("=== DLL依赖关系检查 ===")

    # c10.dll路径（根据你的错误信息）
    c10_path = r"E:\pycharmworkspace\.venv\Lib\site-packages\torch\lib\c10.dll"

    if not os.path.exists(c10_path):
        print(f"❌ c10.dll不存在: {c10_path}")
        return

    print(f"✓ c10.dll存在: {c10_path}")

    # 尝试加载c10.dll
    try:
        c10_dll = ctypes.WinDLL(c10_path)
        print("✓ c10.dll可以正常加载")
        return True
    except Exception as e:
        print(f"❌ c10.dll加载失败: {e}")

        # 检查缺失的依赖
        print("\n=== 检查缺失的DLL ===")
        try:
            result = subprocess.run(['dumpbin', '/dependents', c10_path],
                                    capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print("c10.dll依赖:")
                for line in result.stdout.split('\n'):
                    if '.dll' in line.lower():
                        print(f"  {line.strip()}")
        except:
            print("无法运行dumpbin命令")

        return False


def check_cuda_libraries():
    """检查CUDA相关库"""
    print("\n=== CUDA库检查 ===")

    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Windows\System32",
    ]

    required_dlls = [
        "cudart64_12.dll", "cublas64_12.dll", "cudnn64_8.dll",
        "nvToolsExt64_1.dll", "cufft64_11.dll"
    ]

    for dll in required_dlls:
        found = False
        for path in cuda_paths:
            dll_path = os.path.join(path, dll)
            if os.path.exists(dll_path):
                print(f"✓ {dll}: {dll_path}")
                found = True
                break
        if not found:
            print(f"❌ {dll}: 未找到")


def check_system_environment():
    """检查系统环境"""
    print("\n=== 系统环境检查 ===")

    # 检查Visual C++ Redistributable
    vc_redist_paths = [
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Windows\System32\vcruntime140_1.dll",
        r"C:\Windows\System32\msvcp140.dll",
    ]

    for dll in vc_redist_paths:
        if os.path.exists(dll):
            print(f"✓ {os.path.basename(dll)}: 存在")
        else:
            print(f"❌ {os.path.basename(dll)}: 缺失")


if __name__ == "__main__":
    check_dll_dependencies()
    check_cuda_libraries()
    check_system_environment()