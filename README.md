# Fusing XGB - 积雪水当量融合预测系统

## 项目概述

本项目基于XGBoost算法，融合多源数据（GLDAS、ERA5等）进行积雪水当量(SWE)预测。

## 系统架构

```
数据源 → 预处理 → 特征工程 → 模型训练 → 结果输出
```

## 快速开始

### 本地开发

1. 克隆项目：
```bash
git clone https://github.com/your-username/fusing-xgb.git
cd fusing-xgb
```

2. 安装依赖：
```bash
pip install -r src/requirements.txt
```

3. 运行测试：
```bash
pytest tests/
```

### Docker部署

1. 构建和运行：
```bash
docker-compose up -d
```

2. 访问应用：
- 主应用: http://localhost:8000
- 监控面板: http://localhost:3000 (admin/admin123)

### CI/CD流程

- 推送代码到main分支自动触发部署
- 测试和代码质量检查自动运行
- 自动部署到实验室工作站

## 目录结构

```
fusing-xgb/
├── src/                 # 源代码
├── tests/              # 测试代码
├── docker/             # Docker配置
├── scripts/            # 部署脚本
├── config/             # 配置文件
└── .github/            # GitHub Actions
```

## API文档

### 健康检查
```
GET /health
```

### 系统状态
```
GET /api/status
```

### 最新结果
```
GET /api/results/latest
```

### 文件下载
```
GET /download/{filename}
```

## 监控

系统提供以下监控功能：
- 容器状态监控
- 资源使用情况
- 处理进度跟踪
- 错误日志收集

## 开发指南

### 添加新功能

1. 创建功能分支
2. 实现功能并添加测试
3. 提交Pull Request
4. 通过CI检查后合并

### 部署新版本

推送代码到main分支将自动触发部署流程。

## 故障排除

常见问题及解决方案：

1. **容器启动失败**
   - 检查端口占用
   - 查看Docker日志

2. **数据处理错误**
   - 检查数据文件权限
   - 验证输入数据格式

3. **API无法访问**
   - 检查防火墙设置
   - 验证服务状态
```

### docs/deployment-guide.md
```markdown
# 部署指南

## 实验室工作站设置

### 1. 系统要求
- Ubuntu 20.04+ / CentOS 7+
- Docker 20.10+
- Docker Compose 1.29+
- 至少8GB RAM，50GB存储

### 2. 初始设置

```bash
# 克隆项目
git clone https://github.com/your-username/fusing-xgb.git
cd fusing-xgb

# 运行设置脚本
chmod +x scripts/setup.sh
./scripts/setup.sh

# 配置环境变量
cp .env.example .env
# 编辑.env文件设置您的配置
```

### 3. 手动部署

```bash
# 构建和启动服务
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f app
```

### 4. 自动部署配置

1. 在GitHub仓库设置Secrets：
   - `LAB_HOST`: 实验室工作站IP
   - `LAB_USERNAME`: SSH用户名
   - `LAB_SSH_KEY`: SSH私钥
   - `DOCKER_USERNAME`: Docker Hub用户名
   - `DOCKER_PASSWORD`: Docker Hub密码

2. 推送代码到main分支将自动部署

## 监控和维护

### 日常监控

```bash
# 运行监控脚本
./scripts/monitor.sh

# 检查资源使用
docker stats

# 查看应用日志
docker logs fusing-xgb-app
```

### 备份和恢复

```bash
# 创建备份
./scripts/backup.sh

# 从备份恢复
./scripts/restore.sh backup_file.tar.gz
```

### 故障恢复

1. **服务不可用**
   ```bash
   docker-compose restart
   ```

2. **磁盘空间不足**
   ```bash
   docker system prune -a
   ```

3. **数据损坏**
   ```bash
   ./scripts/restore.sh latest_backup.tar.gz
   ```
```

## 12. 完整的部署流程

### 首次部署步骤：

1. **在实验室工作站上**：
```bash
# 1. 安装必要的软件
sudo apt update && sudo apt install -y git curl

# 2. 克隆项目
git clone https://github.com/your-username/fusing-xgb.git
cd fusing-xgb

# 3. 运行设置脚本
chmod +x scripts/setup.sh
./scripts/setup.sh

# 4. 配置环境变量
cp .env.example .env
# 编辑.env文件，设置您的配置

# 5. 启动服务
docker-compose up -d
```

2. **在GitHub仓库设置Secrets**：
   - 进入仓库 Settings → Secrets → Actions
   - 添加以下secrets：
     - `LAB_HOST`: 您的实验室工作站IP
     - `LAB_USERNAME`: SSH用户名
     - `LAB_SSH_KEY`: SSH私钥内容
     - `DOCKER_USERNAME`: Docker Hub用户名
     - `DOCKER_PASSWORD`: Docker Hub密码
     - `SLACK_WEBHOOK_URL`: (可选) Slack通知webhook

3. **测试部署**：
```bash
# 在实验室工作站上测试
curl http://localhost:8000/health
curl http://localhost:8000/api/status

# 访问监控面板
# http://your-lab-ip:3000 (用户名: admin, 密码: admin123)
```

### 远程访问配置：

为了从外部访问实验室工作站，您可能需要：

1. **配置防火墙**：
```bash
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 3000
```

2. **设置反向代理**（可选）：
```nginx
# 在nginx配置中添加
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
    }
    
    location /monitor/ {
        proxy_pass http://localhost:3000;
    }
}