import sys
# 强制标准输出/标准错误使用 UTF-8 编码并对无法编码的字符安全替换，从根本上解决 Windows 下 Emoji 导致的 UnicodeEncodeError
try:
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 尝试获取我们刚刚设置的测试变量
my_key = os.getenv("TEST_API_KEY")

# 验证结果
if my_key:
    print(f"✅ 环境加载成功！读取到的值为: {my_key}")
else:
    print("❌ 哎呀，没找到变量。请检查 .env 文件是否保存，或者变量名是否拼写正确。")