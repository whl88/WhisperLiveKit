#!/bin/bash

# 检查 whisperlivekit-server 是否已在运行
pids=$(ps aux | grep 'whisperlivekit-server' | grep -v 'grep' | awk '{print $2}')

if [ -n "$pids" ]; then
    echo "检测到 whisperlivekit-server 已在运行："
    ps aux | grep 'whisperlivekit-server' | grep -v 'grep'
    read -p "是否要重启？(Y/n) " answer
    answer=${answer:-Y}
    if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
        echo "操作已取消，未启动新进程。"
        exit 0
    fi
    echo "正在停止已有进程..."
    for pid in $pids; do
        kill -15 $pid
    done
    sleep 2
    # 强制杀死未关闭的进程
    remaining_pids=$(ps -p $pids -o pid=)
    if [ -n "$remaining_pids" ]; then
        for pid in $remaining_pids; do
            kill -9 $pid
        done
    fi
    echo "已停止原有进程，准备启动新进程。"
fi

nohup whisperlivekit-server --model large-v3 --host 0.0.0.0 --port 8081 --ssl-certfile ./cert.pem --ssl-keyfile key.pem --language zh > run.log 2>&1 &
echo "whisperlivekit-server 已启动。"