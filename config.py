"""
config.py — 统一配置管理

解决的问题：
    本地开发用 .env 文件，Streamlit Cloud 用 st.secrets。
    这个模块统一封装密钥获取逻辑，其他模块只需要 from config import get_secret。

优先级：st.secrets > .env 环境变量 > 默认值
"""

import os


def get_secret(key, default=None):
    """
    获取配置项，优先从 st.secrets 读取（云端），
    兜底从环境变量读取（本地 .env）。
    """
    # 1. 优先尝试 Streamlit secrets（云端部署时生效）
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # 2. 兜底：从环境变量读取（本地开发时通过 dotenv 加载）
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv(key, default)


def get_openai_client():
    """
    创建 OpenAI 客户端（复用逻辑，避免每个文件重复写）。
    """
    from openai import OpenAI
    return OpenAI(
        api_key=get_secret("OPENAI_API_KEY"),
        base_url=get_secret("OPENAI_BASE_URL"),
    )
