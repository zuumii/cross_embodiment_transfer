#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${REPO_ROOT}" ]] || { echo "❌ 不在 git 仓库内"; exit 1; }
cd "$REPO_ROOT"

[[ -d models ]] || { echo "❌ 未找到 models/ 目录"; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
# 自定义提交信息可作为第一个参数传入
MSG="${1:-"update(models): $(hostname) $(date -Iseconds)"}"

# 先把远端变更并入本地，减少推送时报错
git fetch --prune
git pull --rebase --autostash origin "$BRANCH"

# 只添加 models/ 的改动
git add -A models

# 没改动就直接退出
if git diff --cached --quiet; then
  echo "ℹ️ models/ 无改动，跳过提交"
  exit 0
fi

git commit -m "$MSG"
git push origin "$BRANCH"
echo "✅ 已推送 models/ 改动到 $BRANCH"