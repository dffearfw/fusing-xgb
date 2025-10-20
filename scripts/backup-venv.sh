#!/bin/bash

set -e

echo "📦 Backing up virtual environment..."

BACKUP_DIR="venv-backup"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份requirements
if [ -f "src/requirements.txt" ]; then
    cp src/requirements.txt $BACKUP_DIR/requirements_$DATE.txt
fi

# 备份已安装的包列表
if [ -d "venv" ]; then
    source venv/bin/activate
    pip freeze > $BACKUP_DIR/requirements_frozen_$DATE.txt
    deactivate
fi

echo "✅ Virtual environment backup completed in $BACKUP_DIR/"