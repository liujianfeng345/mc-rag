"""节点函数"""

from ..utils.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)

from langchain_openai import ChatOpenAI

def create_llm(
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> ChatOpenAI:
    """创建大模型"""
    llm = ChatOpenAI(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    return llm
