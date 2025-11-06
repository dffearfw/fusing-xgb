import os
import ctypes
import subprocess
import sys


def set_dll_directory():
    """设置DLL搜索目录优先级"""
    import sys
    import os

    # 动态查找torch库路径
    torch_lib_path = None
    for path in sys.path:
        potential_path = os.path.join(path, 'torch', 'lib')
        if os.path.exists(potential_path):
            torch_lib_path = potential_path
            break

    if not torch_lib_path:
        # 如果找不到，使用环境变量或默认路径
        torch_lib_path = os.environ.get('TORCH_LIB_PATH',
                                       os.path.join(os.path.dirname(__file__), '..', '..', 'venv', 'Lib', 'site-packages', 'torch', 'lib'))

    # 方法1：使用SetDllDirectory (最高优先级)
    try:
        kernel32 = ctypes.WinDLL('kernel32.dll')
        kernel32.SetDllDirectoryW(torch_lib_path)
        print("✓ 已设置DLL目录优先级")
    except Exception as e:
        print(f"SetDllDirectory失败: {e}")

    # 方法2：强制添加到PATH开头
    os.environ['PATH'] = torch_lib_path + ';' + os.environ['PATH']
    print("✓ 已强制PATH优先级")


def preload_essential_dlls():
    """预加载必需的DLL"""
    torch_lib_path = set_dll_directory()  # 使用动态查找的路径

    print("=== 预加载DLL ===")

    # 按依赖顺序加载
    load_order = [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cudnn64_8.dll",
        "cufft64_11.dll",
        "nvToolsExt64_1.dll",
        "c10.dll"
    ]

    loaded_dlls = {}

    for dll_name in load_order:
        dll_path = os.path.join(torch_lib_path, dll_name)
        if os.path.exists(dll_path):
            try:
                # 使用LoadLibraryEx避免依赖冲突
                kernel32 = ctypes.WinDLL('kernel32.dll')
                handle = kernel32.LoadLibraryExW(dll_path, None, 0x00000800)  # LOAD_WITH_ALTERED_SEARCH_PATH

                if handle:
                    loaded_dlls[dll_name] = handle
                    print(f"✓ 预加载: {dll_name}")
                else:
                    error_code = ctypes.get_last_error()
                    print(f"❌ 预加载失败 {dll_name}: 错误代码 {error_code}")

            except Exception as e:
                print(f"❌ 预加载异常 {dll_name}: {e}")
        else:
            print(f"❌ 文件不存在: {dll_name}")

    return loaded_dlls


def apply_dll_fix_before_import():
    """在导入任何模块之前应用DLL修复"""
    print("=== 应用预导入DLL修复 ===")

    # 设置DLL目录（必须在任何导入之前）
    torch_lib_path = set_dll_directory()  # 使用动态查找的路径

    if os.path.exists(torch_lib_path):
        # 1. 设置进程级DLL目录
        kernel32 = ctypes.WinDLL('kernel32.dll')
        kernel32.SetDllDirectoryW(torch_lib_path)
        print("✓ SetDllDirectoryW 设置成功")

        # 2. 预加载关键DLL（按依赖顺序）
        dll_load_order = [
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "msvcp140.dll",
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cudnn64_8.dll",
            "cufft64_11.dll",
            "c10.dll"
        ]

        for dll_name in dll_load_order:
            dll_path = os.path.join(torch_lib_path, dll_name)
            if os.path.exists(dll_path):
                try:
                    kernel32.LoadLibraryW(dll_path)
                    print(f"✓ 预加载: {dll_name}")
                except Exception as e:
                    print(f"⚠️ 预加载失败 {dll_name}: {e}")

    return torch_lib_path

def setup_environment():
    """设置环境然后启动真正的Python"""

    # 设置DLL路径
    torch_lib_path = set_dll_directory()  # 使用动态查找的路径

    if os.path.exists(torch_lib_path):
        # 设置环境变量
        os.environ['PATH'] = torch_lib_path + ';' + os.environ['PATH']

        # 设置DLL目录
        kernel32 = ctypes.WinDLL('kernel32.dll')
        kernel32.SetDllDirectoryW(torch_lib_path)

        print(f"✓ 环境设置完成: {torch_lib_path}")

    return True


if __name__ == "__main__":
    # 在导入任何其他模块之前修复
    lib_path = apply_dll_fix_before_import()

    print("\n=== 现在导入PyTorch ===")
    try:
        import torch


        print(f"CUDA可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        input("按Enter退出...")