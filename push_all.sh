#!/usr/bin/env bash
set -euo pipefail

### ===== 可配区域（按需改） =====
REMOTE="origin"                 # 远端名
# 固定到某分支就写死：BRANCH="main"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
DEFAULT_MSG="chore: update $(hostname) $(date -Iseconds)"  # 提交说明（可改）
### ===== 可配区域到此 =====

# 允许命令行传入提交说明：./push_all.sh "fix: xxx"
MSG="${1:-$DEFAULT_MSG}"

# 确认在仓库内
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || { echo "❌ 不在 git 仓库内"; exit 1; }
cd "$REPO_ROOT"

# 先同步远端，减少冲突
git fetch --prune "$REMOTE"
git pull --rebase --autostash "$REMOTE" "$BRANCH"

# 添加全部改动（含新增/删除/改名）
git add -A

# 没有变更就退出
if git diff --cached --quiet; then
  echo "ℹ️ 没有需要提交的改动。"
  exit 0
fi

# 提交并推送
git commit -m "$MSG"
git push "$REMOTE" "$BRANCH"
echo "✅ 已推送到 $REMOTE/$BRANCH"