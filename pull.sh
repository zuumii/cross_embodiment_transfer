#!/usr/bin/env bash
set -euo pipefail

# 进入仓库根
REPO_ROOT="$(git rev-parse --show-toplevel)"; cd "$REPO_ROOT"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# 1) 保存快照分支 + 备份工作区（含未跟踪文件）
SNAP="pre-override-$(date +%Y%m%d-%H%M%S)"
git branch "$SNAP" >/dev/null 2>&1 || true          # 记下当前 HEAD
git stash push -u -m "$SNAP" >/dev/null 2>&1 || true # 备份工作区/未跟踪

# 2) 以远端为准覆盖本地
git fetch --prune
git reset --hard "origin/$BRANCH"    # 丢弃本地对已跟踪文件的改动
git clean -fd                         # 清理未跟踪文件/目录（保留被 .gitignore 忽略的）

echo "✅ 已用远端覆盖本地：$BRANCH"
echo "🛟 可回滚："
echo "   # 回到覆盖前的提交："
echo "   git reset --hard $SNAP"
echo "   # 或从 stash 恢复工作区改动："
echo "   git stash list | grep $SNAP && git stash apply \"stash^{/$SNAP}\""