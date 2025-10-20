.PHONY: help venv setup test deploy clean monitor

help:
	@echo "Available commands:"
	@echo "  venv     - Create and setup virtual environment"
	@echo "  setup    - Setup development environment"
	@echo "  test     - Run tests in virtual environment"
	@echo "  deploy   - Deploy to lab workstation"
	@echo "  clean    - Clean up resources"
	@echo "  monitor  - Show system status"

venv:
	@echo "🐍 Setting up virtual environment..."
	chmod +x scripts/setup-venv.sh
	./scripts/setup-venv.sh

setup: venv
	@echo "🔧 Setting up development environment..."
	source venv/bin/activate && pip install -r requirements-dev.txt

test:
	@echo "🧪 Running tests..."
	source venv/bin/activate && pytest tests/ -v

build:
	@echo "🐳 Building Docker image..."
	docker build -t fusing-xgb:latest .

deploy:
	@echo "🚀 Deploying to lab workstation..."
	chmod +x scripts/deploy.sh
	./scripts/deploy.sh

clean:
	@echo "🧹 Cleaning up..."
	docker system prune -f
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__

clean-venv:
	@echo "🧹 Removing virtual environment..."
	rm -rf venv

monitor:
	@echo "📊 System monitor..."
	chmod +x scripts/monitor.sh
	./scripts/monitor.sh

backup-venv:
	@echo "📦 Backing up virtual environment..."
	chmod +x scripts/backup-venv.sh
	./scripts/backup-venv.sh