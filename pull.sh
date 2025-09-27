#!/usr/bin/env bash
set -euo pipefail

# 确认在 git 仓库内
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${REPO_ROOT}" ]] || { echo "❌ 不在 git 仓库内"; exit 1; }
cd "$REPO_ROOT"

# 当前分支（默认）
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# 拉最新 + 自动把本地未提交改动临时保存再还原（避免冲突卡住）
git fetch --prune
git pull --rebase --autostash origin "$BRANCH"

echo "✅ 已拉取最新：$BRANCH @ $REPO_ROOT"