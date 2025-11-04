FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
# 设置工作目录（容器内的文件夹）
WORKDIR /workspace

# 复制你的整个项目到容器中
COPY . .

# 安装可能需要的其他Python包（如果有的话）
# RUN pip install pandas scikit-learn matplotlib

# 设置Python路径，让Python能找到你的代码
ENV PYTHONPATH=/workspace

# 运行你的CUDA测试脚本
CMD ["python", "src/training/check_cuda.py"]