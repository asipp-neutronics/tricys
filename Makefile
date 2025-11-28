# ------------------------------------------------------------------------------
# Makefile for Python Project
#
.PHONY: help clean install dev-install lint format check test docs-install docs-serve docs-build

# 默认目标：当只输入 `make` 时，会执行此命令，显示帮助信息。
help:
	@echo "Usage: make <command>"
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install the project in editable mode for regular use."
	@echo "  dev-install   Install the project with all development dependencies."
	@echo "  docs-install  Install dependencies for building documentation."
	@echo "  docs-serve    Serve the documentation site locally for development."
	@echo "  docs-build    Build the documentation site."
	@echo "  clean         Remove all build artifacts, cache files, and logs."
	@echo "  lint          Check code style and potential errors (report only, do not modify)."
	@echo "  format        Automatically format and repair code."
	@echo "  check         Combine commands: format first, then check to make sure the codebase is clean."
	@echo "  test          Perform one-click tests."
	@echo "  help          Show this help message."

# 安装项目的核心依赖，用于常规使用或部署。
# -e 表示以“可编辑模式”安装，你对源代码的修改会立刻生效，无需重新安装。
install:
	@echo "--> Installing project in editable mode..."
	pip install -e .
	omc ./script/modelica_install/install.mos 
	@echo "--> Installation complete."

# 安装项目的所有依赖，包括开发和测试工具。
# ".[dev]" 语法会读取 pyproject.toml 中 [project.optional-dependencies] 下的 dev 分组。
dev-install:
	@echo "--> Installing project with development dependencies..."
	pip install -e ".[dev]"
	omc ./script/modelica_install/install.mos 
	@echo "--> Development installation complete."

# Install documentation dependencies
docs-install:
	@echo "--> Installing documentation dependencies..."
	pip install -e ".[docs]"
	@echo "--> Documentation dependencies installed."

# Serve the documentation site locally
docs-serve:
	@echo "--> Starting local documentation server..."
	mkdocs serve

# Build the documentation site
docs-build:
	@echo "--> Building documentation..."
	mkdocs build


# 清理项目，删除所有自动生成的文件和目录。
clean:
	@echo "--> Cleaning up project..."
	# 使用 find 命令安全地查找并删除所有 __pycache__ 目录
	find . -type d -name "__pycache__" -exec rm -rf {} +
	# 删除所有 .pyc 编译文件
	find . -type f -name "*.pyc" -delete
	# 删除其他常见的缓存、构建和日志目录/文件
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf tricys/*.egg-info/
	rm -f .coverage
	rm -rf temp/
	rm -rf log/
	rm -rf results/
	rm -rf data/
	rm -rf test/test_*/
	@echo "--> Cleanup complete."

# 检查代码风格和潜在错误（只报告，不修改）
lint:
	@echo "--> Checking code with Ruff..."
	ruff check .

# 自动格式化和修复代码
format:
	@echo "--> Formatting code with Black..."
	black .
	@echo "--> Sorting imports and fixing code with Ruff..."
	ruff check . --fix
	@echo "--> Code formatting complete."

# 组合命令：先格式化，再检查，确保代码库是干净的
check: format lint
	@echo "--> All checks passed!"

# 执行一键测试
test:
	@echo "--> Pytest Project..."
	pytest -v test/.
