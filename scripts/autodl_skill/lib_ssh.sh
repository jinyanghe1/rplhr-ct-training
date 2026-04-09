#!/bin/bash
#===============================================================================
# SSH 连接工具库 - 提供稳定可靠的SSH连接功能
# 用法: source $0
#===============================================================================

# 默认配置
SSH_HOST="${AUTODL_HOST:-connect.westd.seetacloud.com}"
SSH_PORT="${AUTODL_PORT:-23086}"
SSH_USER="${AUTODL_USER:-root}"
SSH_PASS="${AUTODL_PASS:-}"

# SSH连接参数
SSH_OPTS="-o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o BatchMode=yes \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o ExitOnForwardFailure=yes"

# 重试配置
MAX_RETRIES=5
RETRY_DELAY=3

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { [[ "$DEBUG" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $1"; }

#===============================================================================
# 检查SSH连接是否可用
#===============================================================================
ssh_check_connection() {
    local host="${1:-$SSH_HOST}"
    local port="${2:-$SSH_PORT}"
    local user="${3:-$SSH_USER}"
    
    ssh $SSH_OPTS -p "$port" "$user@$host" "echo 'SSH_OK'" 2>/dev/null | grep -q "SSH_OK"
    return $?
}

#===============================================================================
# 带重试的SSH执行 - 核心函数
# 参数: $1=命令, $2=超时(秒,可选), $3=重试次数(可选)
#===============================================================================
ssh_run_with_retry() {
    local cmd="$1"
    local timeout_sec="${2:-60}"
    local max_retries="${3:-$MAX_RETRIES}"
    local delay="$RETRY_DELAY"
    
    log_debug "执行命令: ${cmd:0:80}..."
    
    for i in $(seq 1 $max_retries); do
        local output
        local exit_code
        
        # 使用SSH内置超时机制
        output=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
            export PATH=\$PATH:/usr/local/bin:/usr/bin:/bin
            $cmd
        " 2>&1)
        exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            echo "$output"
            return 0
        fi
        
        # 检查特定的错误类型
        if echo "$output" | grep -qi "connection refused"; then
            log_error "SSH连接被拒绝，检查服务器状态"
            return 1
        elif echo "$output" | grep -qi "connection timed out"; then
            log_warn "SSH连接超时，尝试重试 ($i/$max_retries)..."
        elif echo "$output" | grep -qi "permission denied"; then
            log_error "SSH权限被拒绝，检查密钥配置"
            return 1
        else
            log_warn "SSH执行失败 (exit=$exit_code)，重试 ($i/$max_retries)..."
        fi
        
        if [[ $i -lt $max_retries ]]; then
            sleep $delay
            # 指数退避
            delay=$((delay * 2))
            [[ $delay -gt 30 ]] && delay=30
        fi
    done
    
    log_error "SSH命令执行失败，已重试 $max_retries 次"
    echo "$output"
    return 1
}

#===============================================================================
# 使用expect的SSH执行（用于密码认证）
#===============================================================================
ssh_run_with_expect() {
    local cmd="$1"
    local host="${2:-$SSH_HOST}"
    local port="${3:-$SSH_PORT}"
    local user="${4:-$SSH_USER}"
    local pass="${5:-$SSH_PASS}"
    
    # 检查expect是否安装
    if ! command -v expect &>/dev/null; then
        log_error "expect 未安装，无法使用密码认证"
        return 1
    fi
    
    expect << EOF
        set timeout 60
        log_user 0
        spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port $user@$host "$cmd"
        expect {
            "password:" {
                send "$pass\r"
                exp_continue
            }
            "yes/no" {
                send "yes\r"
                exp_continue
            }
            eof
        }
        catch wait result
        puts "\$result"
EOF
}

#===============================================================================
# 检查训练进程是否在运行
#===============================================================================
is_training_running() {
    local process_name="${1:-trainxuanwu}"
    local output
    
    output=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
        ps aux | grep '[${process_name:0:1}]${process_name:1}' | grep -v grep | wc -l
    " 2>/dev/null)
    
    local count=$(echo "$output" | grep -oE '[0-9]+' | tail -1 || echo "0")
    [[ "$count" -gt 0 ]]
    return $?
}

#===============================================================================
# 获取训练进程PID
#===============================================================================
get_training_pid() {
    local process_name="${1:-trainxuanwu}"
    local pid
    
    pid=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
        ps aux | grep '[${process_name:0:1}]${process_name:1}' | grep -v grep | awk '{print \$2}' | head -1
    " 2>/dev/null)
    
    echo "$pid"
}

#===============================================================================
# 获取训练进程运行时间
#===============================================================================
get_training_uptime() {
    local pid="$1"
    local uptime
    
    uptime=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
        ps -p $pid -o etime= 2>/dev/null | tr -d ' '
    " 2>/dev/null)
    
    echo "${uptime:-N/A}"
}

#===============================================================================
# 获取GPU使用情况
#===============================================================================
get_gpu_usage() {
    local pid="${1:-}"
    local gpu_info
    
    if [[ -n "$pid" ]]; then
        gpu_info=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
            nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | grep $pid
        " 2>/dev/null)
    else
        gpu_info=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
            nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
        " 2>/dev/null)
    fi
    
    echo "${gpu_info:-N/A}"
}

#===============================================================================
# 下载远程文件到本地
#===============================================================================
ssh_download_file() {
    local remote_path="$1"
    local local_path="$2"
    
    if ! ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "test -f $remote_path" 2>/dev/null; then
        log_error "远程文件不存在: $remote_path"
        return 1
    fi
    
    ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "cat $remote_path" > "$local_path" 2>/dev/null
    return $?
}

#===============================================================================
# 测试SSH连接
#===============================================================================
ssh_test() {
    echo "测试SSH连接..."
    echo "  主机: $SSH_HOST:$SSH_PORT"
    echo "  用户: $SSH_USER"
    
    if ssh_check_connection; then
        log_info "SSH连接成功!"
        
        # 获取远程系统信息
        local sys_info
        sys_info=$(ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "
            echo \"OS: \$(uname -s) \$(uname -r)\"
            echo \"GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')\"
        " 2>/dev/null)
        echo ""
        echo "远程系统信息:"
        echo "$sys_info" | sed 's/^/  /'
        return 0
    else
        log_error "SSH连接失败!"
        echo ""
        echo "请检查:"
        echo "  1. SSH密钥是否正确配置 (ssh-copy-id)"
        echo "  2. 服务器地址和端口是否正确"
        echo "  3. 网络连接是否正常"
        return 1
    fi
}
