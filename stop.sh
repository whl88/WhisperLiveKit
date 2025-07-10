#!/bin/bash

# 通过进程名查找进程ID
pids=$(ps aux | grep 'whisperlivekit-server' | grep -v 'grep' | awk '{print $2}')

# 如果没有找到进程，就输出提示并退出
if [ -z "$pids" ]; then
    echo "没有找到包含关键字 'whisperlivekit-server' 的进程"
    exit 0
fi

echo "找到以下进程："
ps aux | grep 'whisperlivekit-server' | grep -v 'grep'

# 询问用户是否要杀死这些进程
read -p "确定要杀死这些进程吗？(Y/n) " answer
answer=${answer:-Y}
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    echo "操作已取消"
    exit 0
fi

# 先尝试使用 SIGTERM(15) 信号优雅地终止进程
echo "正在发送 SIGTERM 信号..."
for pid in $pids; do
    kill -15 $pid
done

# 等待一段时间，让进程有机会优雅地关闭
sleep 2

# 检查进程是否还在运行
remaining_pids=$(ps -p $pids -o pid=)
if [ -n "$remaining_pids" ]; then
    echo "进程仍在运行，正在发送 SIGKILL(9) 信号强制终止..."
    for pid in $remaining_pids; do
        kill -9 $pid
    done
    echo "已发送 SIGKILL 信号"
else
    echo "进程已成功终止"
fi

# 验证进程是否已终止
final_check=$(ps aux | grep 'whisperlivekit-server' | grep -v 'grep')
if [ -z "$final_check" ]; then
    echo "所有包含关键字 'whisperlivekit-server' 的进程都已成功终止"
else
    echo "以下进程仍然存在："
    echo "$final_check"
fi