FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY src/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ .

# 创建必要的目录
RUN mkdir -p /app/outputs/combined \
    /app/outputs/cswe/last \
    /app/outputs/era5_swe \
    /app/outputs/era5_temperature \
    /app/outputs/glsnow \
    /app/outputs/landcover \
    /app/outputs/landuse \
    /app/outputs/snow_depth/temp \
    /app/outputs/snow_phenology \
    /app/outputs/terrain_features \
    /app/outputs/train_results \
    /app/process/logs \
    /app/download/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "process.main"]