#!/bin/bash

# 要运行的命令
CMD="conda run -n ma python train_rl.py --config configs/Reach/TD3_JV.yml"

# 同时保持的任务数
MAX_JOBS=10

while true; do
    # 当前运行的任务数
    RUNNING=$(jobs -r | wc -l)

    # 如果运行的任务数少于 MAX_JOBS，就补充新的任务
    while [ $RUNNING -lt $MAX_JOBS ]; do
        echo "启动新任务..."
        $CMD &
        RUNNING=$(jobs -r | wc -l)
        sleep 2   # 避免瞬间启动过快
    done

    # 每隔 5 秒检查一次
    sleep 5
done
