#!/bin/bash

# 监控脚本
echo "📊 Fusing XGB System Monitor"
echo "=============================="

# 检查容器状态
echo "🐳 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""

# 检查资源使用
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""

# 检查日志
echo "📝 Recent Logs:"
docker logs fusing-xgb-app --tail 10 2>/dev/null || echo "No logs available"

echo ""

# 检查磁盘空间
echo "💿 Disk Space:"
df -h /opt/fusing-xgb

echo ""

# API健康检查
echo "🏥 API Health:"
curl -s http://localhost:8000/health || echo "API not available"