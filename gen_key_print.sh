#!/usr/bin/env bash
set -e

# 可选参数：自定义标签（默认 models），会体现在文件名和注释里
LABEL="${1:-models}"

SSH_DIR="$HOME/.ssh"
KEY="$SSH_DIR/id_ed25519_${LABEL}"
PUB="${KEY}.pub"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# 已存在就不覆盖，直接把公钥打印出来
if [[ -f "$PUB" ]]; then
  echo "ℹ️ 已存在公钥：$PUB"
  echo "----- 公钥开始 -----"
  cat "$PUB"
  echo "----- 公钥结束 -----"
  exit 0
fi

# 生成密钥（无口令），备注包含用户名@主机名和自定义标签
ssh-keygen -t ed25519 -C "${USER}@$(hostname) ${LABEL}" -f "$KEY" -N "" >/dev/null

chmod 600 "$KEY"
chmod 644 "$PUB"

echo "✅ 生成完成。请复制下面这一整行公钥："
echo "----- 公钥开始 -----"
cat "$PUB"
echo "----- 公钥结束 -----"