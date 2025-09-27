sudo tee /usr/local/bin/reboot-on-ram.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

THRESHOLD_GB=60
LOCK=/run/reboot-on-ram.lock   # 10 分钟防抖
LOG=/var/log/reboot-on-ram.log

# 用 MemAvailable 更接近真实可用（排除缓存）
read -r MT MA < <(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}' /proc/meminfo)
used_kb=$(( MT - MA ))
used_gb=$(( (used_kb + 1048575) / 1048576 ))   # kB -> GiB，向上取整

# 只在超阈值时记录一行日志&尝试重启
if (( used_gb >= THRESHOLD_GB )); then
  # 10 分钟内只触发一次，避免连环重启
  if [[ ! -e "$LOCK" ]] || (( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) >= 600 )); then
    touch "$LOCK"
    echo "$(date -Is) used=${used_gb}GB >= ${THRESHOLD_GB}GB → reboot" >> "$LOG"
    logger -t reboot-on-ram "RAM used ${used_gb}GB >= ${THRESHOLD_GB}GB, rebooting"
    systemctl reboot
  fi
fi
EOF
sudo chmod +x /usr/local/bin/reboot-on-ram.sh