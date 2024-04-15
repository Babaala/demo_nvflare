#!/bin/bash

# 定义端口数组
ports=(8102 8103)

# 遍历端口并处理每个端口
for port in "${ports[@]}"
do
    echo "Checking for processes on port $port..."
    # 使用 lsof 获取占用指定端口的进程 PID
    pids=$(lsof -ti :$port)

    # 检查是否有 PID 返回
    if [ -z "$pids" ]; then
        echo "No process found on port $port."
    else
        # 如果找到 PID，则终止这些进程
        echo "Killing processes on port $port with PIDs: $pids"
        kill -9 $pids
    fi
done