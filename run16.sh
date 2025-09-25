#!/usr/bin/env bash
set -euo pipefail

# ====== 配置项 ======
CMD='conda run -n ma python train_align.py --config configs/Reach/align_JV.yml'
MAX_JOBS=10          # 并发数量，按需改 10/12 都行
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${ROOT}/run_logs"
mkdir -p "$LOGDIR"
# ====================

pids=()
last_sec=""

cleanup() {
  echo; echo ">> 停止所有训练进程..."
  if ((${#pids[@]})); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    # 仍存活就强杀
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
    done
  fi
  echo ">> 已清理。"
}
trap cleanup INT TERM HUP

# 等到下一秒，确保每个任务的 HH-MM-SS 唯一
wait_next_second() {
  local cur
  cur="$(date +%H-%M-%S)"
  if [[ "$cur" == "$last_sec" ]]; then
    # 睡到秒跳变
    while [[ "$(date +%H-%M-%S)" == "$cur" ]]; do sleep 0.1; done
  fi
  last_sec="$(date +%H-%M-%S)"
}

start_one() {
  wait_next_second
  local ts day log
  ts="$(date +%H-%M-%S)"
  day="$(date +%m.%d.%Y)"
  # 预创建当天 logs 目录（你的代码也会创建，这里只是降低竞态）
  mkdir -p "${ROOT}/logs/${day}"

  log="${LOGDIR}/run_${day}_${ts}_$$.log"
  echo "[${ts}] 启动任务，日志：${log}"
  (
    cd "$ROOT"
    # -oL/-eL 让输出行缓冲，tail 实时看
    stdbuf -oL -eL bash -lc "$CMD"
  ) >"$log" 2>&1 &
  pids+=($!)
}

# 主循环：维持 MAX_JOBS 并发
echo "并发数: ${MAX_JOBS}"
echo "Ctrl-C 可停止并清理全部训练"
while true; do
  # 刷新存活 PID 列表
  alive=()
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); fi
  done
  pids=("${alive[@]}")

  # 补足并发
  while ((${#pids[@]} < MAX_JOBS)); do
    start_one
    sleep 0.5
  done

  sleep 2
done