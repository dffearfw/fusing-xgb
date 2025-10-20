#!/bin/bash

set -e

echo "🔧 Setting up Python virtual environment..."

# 检查是否已存在venv
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "📦 Virtual environment already exists, skipping creation..."
else
    echo "🐍 Creating new virtual environment..."
    python -m venv venv
fi

# 激活虚拟环境
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# 升级pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 Installing dependencies..."
if [ -f "src/requirements.txt" ]; then
    pip install -r src/requirements.txt
else
    echo "❌ requirements.txt not found in src directory"
    exit 1
fi

# 检查是否有额外的依赖
if [ -f "requirements-dev.txt" ]; then
    echo "📚 Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

echo "✅ Virtual environment setup completed!"
echo ""
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"