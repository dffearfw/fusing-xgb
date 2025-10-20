#!/bin/bash

set -e

echo "🚀 Starting deployment..."

# 配置变量
PROJECT_DIR="/opt/fusing-xgb"
BACKUP_DIR="/opt/backups/fusing-xgb"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份
echo "📦 Creating backup..."
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz $PROJECT_DIR/data $PROJECT_DIR/outputs 2>/dev/null || true

# 停止现有服务
echo "🛑 Stopping existing services..."
cd $PROJECT_DIR
docker-compose down

# 清理旧镜像
echo "🧹 Cleaning up..."
docker system prune -f

# 更新代码
echo "📥 Pulling latest changes..."
git pull origin main

# 更新镜像
echo "🐳 Pulling latest Docker images..."
docker-compose pull

# 启动服务
echo "🚀 Starting services..."
docker-compose up -d

# 等待服务启动
echo "⏳ Waiting for services to start..."
sleep 30

# 健康检查
echo "🏥 Performing health check..."
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed"
    exit 1
}

echo "✅ Deployment completed successfully!"