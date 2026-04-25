"""
RAG 系统配置管理

使用环境变量和 .env 文件统一管理配置
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# LLM 配置 - 使用 DeepSeek API（兼容 OpenAI SDK）
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# LLM 推理参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# =============================================================================
# 向量数据库配置
# =============================================================================
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(PROJECT_ROOT / "chroma_db"))
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "mc_docs")

# 检索参数
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
