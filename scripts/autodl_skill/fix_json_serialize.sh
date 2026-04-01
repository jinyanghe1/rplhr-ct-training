#!/bin/bash
#===============================================================================
# 修复 trainxuanwu.py 中的 JSON 序列化问题
# 将 numpy float32 转换为 Python float
#===============================================================================

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}[INFO]${NC} 修复 JSON 序列化问题..."

expect << 'EOF'
set timeout 30
spawn ssh -o StrictHostKeyChecking=no -p 23086 root@connect.westd.seetacloud.com
expect "password:"
send "Z9wdTD/ZA6fZ\r"
expect "# "
send "cd /root/autodl-tmp/rplhr-ct-training-main/code\r"
expect "# "
send "sed -i 's/json.dump(history, f, indent=2)/import json\\n        def convert_to_json(obj):\\n            if isinstance(obj, dict):\\n                return {k: convert_to_json(v) for k, v in obj.items()}\\n            elif isinstance(obj, list):\\n                return [convert_to_json(i) for i in obj]\\n            elif hasattr(obj, 'item'):\\n                return obj.item()\\n            elif hasattr(obj, '__float__'):\\n                return float(obj)\\n            return obj\\n        json.dump(convert_to_json(history), f, indent=2)/g' trainxuanwu.py\r"
expect "# "
send "grep -A 15 'Training completed' trainxuanwu.py | head -20\r"
expect "# "
send "exit\r"
expect eof
EOF

echo -e "${GREEN}[DONE]${NC} 修复完成"
