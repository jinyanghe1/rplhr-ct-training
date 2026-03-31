#!/usr/bin/env python3
"""修复 trainxuanwu.py 的 JSON 序列化问题"""

import sys

filepath = '/root/autodl-tmp/rplhr-ct-training-main/code/trainxuanwu.py'

with open(filepath, 'r') as f:
    content = f.read()

# 检查是否已经修复
if 'make_serializable' in content:
    print("Already fixed")
    sys.exit(0)

# 目标代码
old_code = "        json.dump(history, f, indent=2)"

new_code = """        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, '__float__'):
                return float(obj)
            return obj
        json.dump(make_serializable(history), f, indent=2)"""

if old_code not in content:
    print(f"Could not find target code: {old_code}")
    sys.exit(1)

content = content.replace(old_code, new_code)

with open(filepath, 'w') as f:
    f.write(content)

print("Fixed!")
