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