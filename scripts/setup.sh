#!/bin/bash

set -e

echo "🔧 Setting up Fusing XGB project..."

# 创建目录
mkdir -p /opt/fusing-xgb/{data,logs,outputs,config}

# 复制配置文件
cp -r config/* /opt/fusing-xgb/config/

# 设置权限
chmod -R 755 /opt/fusing-xgb
chown -R $USER:$USER /opt/fusing-xgb

# 安装Docker（如果未安装）
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker $USER
fi

# 安装Docker Compose（如果未安装）
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

echo "✅ Setup completed!"