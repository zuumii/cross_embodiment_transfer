#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${REPO_ROOT}" ]] || { echo "❌ 不在 git 仓库内"; exit 1; }
cd "$REPO_ROOT"

[[ -d models ]] || { echo "❌ 未找到 models/ 目录"; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MSG="${1:-"update(models): $(hostname) $(date -Iseconds)"}"

# 先同步远端，减少推送时报错（不会 push）
git fetch --prune
git pull --rebase --autostash origin "$BRANCH"

# 关键：清空暂存区，只保留工作区改动（防止误提交其它路径）
git reset

# 只把 models/ 的新增/修改/删除放入暂存区
git add -A -- models

# 若 models/ 没变化就退出
if git diff --cached --quiet; then
  echo "ℹ️ models/ 无改动，跳过提交"
  exit 0
fi

git commit -m "$MSG"
git push origin "$BRANCH"
echo "✅ 已推送 models/ 改动到 $BRANCH"