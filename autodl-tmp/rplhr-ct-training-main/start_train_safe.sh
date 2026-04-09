#!/bin/bash
# RPLHR-CT 训练启动脚本（可靠守护版）
# 使用 systemd 用户级服务，确保训练进程不会因 SSH 断开而终止
#
# 用法:
#   bash start_train_safe.sh              # 启动训练
#   bash start_train_safe.sh status       # 查看训练状态
#   bash start_train_safe.sh log          # 查看训练日志
#   bash start_train_safe.sh stop         # 停止训练
#   bash start_train_safe.sh restart      # 重启训练

set -e

SERVICE_NAME="train-rplhr"
SERVICE_FILE="$HOME/.config/systemd/user/${SERVICE_NAME}.service"
LOG_FILE="/root/autodl-tmp/rplhr-ct-training-main/train_$(date +%Y%m%d).log"
CODE_DIR="/root/autodl-tmp/rplhr-ct-training-main/code"

# ==================== 训练参数配置 ====================
NET_IDX="xuanwu_ratio4_200epoch"
PATH_KEY="dataset01_xuanwu"
EPOCHS=200
NUM_WORKERS=2
TEST_NUM_WORKERS=2
# =====================================================

create_service() {
    mkdir -p "$HOME/.config/systemd/user/"
    
    # 检测Python路径
    PYTHON_BIN=$(which python3 || which python)
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=RPLHR-CT Super-Resolution Training (${NET_IDX})
After=network.target

[Service]
Type=simple
WorkingDirectory=${CODE_DIR}
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"
ExecStart=${PYTHON_BIN} trainxuanwu.py train \\
    --net_idx="${NET_IDX}" \\
    --path_key="${PATH_KEY}" \\
    --epoch=${EPOCHS} \\
    --use_augmentation=True \\
    --normalize_ct=True \\
    --window_center=40 \\
    --window_width=400 \\
    --num_workers=${NUM_WORKERS} \\
    --test_num_workers=${TEST_NUM_WORKERS}
Restart=on-failure
RestartSec=60
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}

[Install]
WantedBy=default.target
EOF

    echo "✅ 服务文件已创建: $SERVICE_FILE"
    systemctl --user daemon-reload
}

start_training() {
    echo "========================================"
    echo "🚀 RPLHR-CT 训练 - 安全守护模式"
    echo "📊 配置: ${NET_IDX}"
    echo "📅 时间: $(date)"
    echo "📁 日志: ${LOG_FILE}"
    echo "========================================"
    
    # 预检查
    echo ""
    echo "📋 预检查:"
    
    # 检查磁盘空间
    DISK_AVAIL=$(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')
    echo "  磁盘可用: ${DISK_AVAIL}"
    
    # 检查内存
    MEM_AVAIL=$(free -h | grep Mem | awk '{print $7}')
    echo "  内存可用: ${MEM_AVAIL}"
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "N/A")
        echo "  GPU: ${GPU_INFO}"
    fi
    
    # 检查是否已有训练在运行
    if systemctl --user is-active "${SERVICE_NAME}" &>/dev/null; then
        echo ""
        echo "⚠️  训练服务已在运行中!"
        echo "  使用 '$0 status' 查看状态"
        echo "  使用 '$0 stop' 停止训练"
        return 1
    fi
    
    # 检查已有的模型权重
    MODEL_DIR="/root/autodl-tmp/rplhr-ct-training-main/model/${PATH_KEY}/${NET_IDX}"
    if [ -d "$MODEL_DIR" ]; then
        echo "  ⚠️ 模型目录已存在: ${MODEL_DIR}"
        echo "  已有权重文件:"
        ls -la "$MODEL_DIR"/*.pkl 2>/dev/null || echo "    (无)"
    fi
    
    echo ""
    
    # 创建服务
    create_service
    
    # 启动服务
    systemctl --user start "${SERVICE_NAME}"
    sleep 2
    
    # 检查是否成功启动
    if systemctl --user is-active "${SERVICE_NAME}" &>/dev/null; then
        echo "✅ 训练已启动! (systemd守护)"
        echo ""
        echo "📡 监控命令:"
        echo "  查看状态: $0 status"
        echo "  实时日志: $0 log"
        echo "  停止训练: $0 stop"
        echo "  重启训练: $0 restart"
        echo ""
        echo "💡 现在可以安全断开SSH，训练将继续运行!"
    else
        echo "❌ 启动失败! 查看日志:"
        journalctl --user -u "${SERVICE_NAME}" -n 20
        return 1
    fi
}

stop_training() {
    echo "🛑 停止训练..."
    systemctl --user stop "${SERVICE_NAME}"
    echo "✅ 训练已停止"
    
    # 显示最后的训练进度
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📊 最后训练进度:"
        tail -20 "$LOG_FILE" | grep -E "(Epoch|PSNR|Train Loss|Best)" || tail -5 "$LOG_FILE"
    fi
}

show_status() {
    echo "📊 训练服务状态:"
    systemctl --user status "${SERVICE_NAME}" 2>/dev/null || echo "  服务未运行"
    
    echo ""
    echo "📈 最新训练进度:"
    if [ -f "$LOG_FILE" ]; then
        tail -30 "$LOG_FILE" | grep -E "(Epoch|PSNR|Train Loss|Best|Summary)" || tail -10 "$LOG_FILE"
    else
        echo "  日志文件不存在: ${LOG_FILE}"
    fi
    
    echo ""
    echo "⏱️  训练运行时间:"
    systemctl --user show "${SERVICE_NAME}" --property=ActiveEnterTimestamp 2>/dev/null || echo "  未知"
}

show_log() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "日志文件不存在: ${LOG_FILE}"
        journalctl --user -u "${SERVICE_NAME}" -f
    fi
}

restart_training() {
    echo "🔄 重启训练..."
    systemctl --user restart "${SERVICE_NAME}"
    sleep 2
    show_status
}

# ==================== 主入口 ====================
case "${1:-start}" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    status)
        show_status
        ;;
    log)
        show_log
        ;;
    restart)
        restart_training
        ;;
    *)
        echo "用法: $0 {start|stop|status|log|restart}"
        exit 1
        ;;
esac
