#!/usr/bin/env bash
set -euo pipefail

THRESHOLD_GB=60
LOCK=/run/reboot-on-ram.lock   # 10 分钟防抖
LOG=/var/log/reboot-on-ram.log

# 读取内存，已用 = MemTotal - MemAvailable（更贴近真实可用）
read -r MT MA < <(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}' /proc/meminfo)
used_kb=$(( MT - MA ))
threshold_kb=$(( THRESHOLD_GB * 1048576 ))   # 1 GiB = 1048576 kB

# 只在超阈值时记日志并重启
if (( used_kb >= threshold_kb )); then
  # 10 分钟内只触发一次，避免连环重启
  last=$(stat -c %Y "$LOCK" 2>/dev/null || echo 0)
  if (( $(date +%s) - last >= 600 )); then
    touch "$LOCK"
    used_gb=$(( used_kb / 1048576 ))
    echo "$(date -Is) used=${used_gb}GB >= ${THRESHOLD_GB}GB → reboot" >> "$LOG"
    logger -t reboot-on-ram "RAM used ${used_gb}GB >= ${THRESHOLD_GB}GB, rebooting"
    systemctl reboot
  fi
fi