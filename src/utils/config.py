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

# 混合检索参数
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))   # BM25 关键词检索返回数量
RRF_K = int(os.getenv("RRF_K", "60"))             # RRF 重排序平滑参数


# =============================================================================
# 嵌入模型配置
# 使用 HuggingFace 本地嵌入（多语言模型，支持中文）
# =============================================================================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # 如有GPU改为 cuda


# =============================================================================
# 文档处理配置
# =============================================================================
DOCS_DIR = os.getenv("DOCS_DIR", str(PROJECT_ROOT / "data"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# =============================================================================
# LangSmith 追踪配置
# 设置环境变量后，LangChain/LangGraph 会自动上报 trace 到 LangSmith
# 官网: https://smith.langchain.com
# =============================================================================
LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "mc-rag")

# 文档分块分隔符（Markdown 特有）
CHUNK_SEPARATORS = [
    "\n## ",     # 二级标题优先
    "\n### ",    # 三级标题
    "\n#### ",   # 四级标题
    "\n---\n",   # 水平分割线
    "\n\n",      # 段落
    "\n",        # 普通换行
    " ",         # 最后按空格
]
