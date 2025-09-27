#!/usr/bin/env bash
set -euo pipefail

# ===== 可配/可覆盖的参数 =====
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
DEFAULT_MSG="${DEFAULT_MSG:-chore: update $(hostname) $(date -Iseconds)}"
TRACKED_ONLY="${TRACKED_ONLY:-0}"   # 1=只提交已跟踪文件改动（不包含新文件）
SKIP_PULL="${SKIP_PULL:-0}"         # 1=不执行 pull
# ===== 以上可通过环境变量覆盖 =====

MSG="${1:-$DEFAULT_MSG}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || { echo "❌ 不在 git 仓库内"; exit 1; }
cd "$REPO_ROOT"

# 先同步远端（可跳过）
if [[ "$SKIP_PULL" -ne 1 ]]; then
  git fetch --prune "$REMOTE"
  git pull --rebase --autostash "$REMOTE" "$BRANCH"
fi

# 清空暂存区，避免历史残留
git reset

# 选择性地收集变更
if [[ "$TRACKED_ONLY" -eq 1 ]]; then
  git add -u        # 只已跟踪文件的修改/删除
else
  git add -A        # 所有改动（新增/修改/删除）
fi

# 无改动直接退出
if git diff --cached --quiet; then
  echo "ℹ️ 没有需要提交的改动。"
  exit 0
fi

git commit -m "$MSG"
git push "$REMOTE" "$BRANCH"
echo "✅ 已推送到 $REMOTE/$BRANCH"