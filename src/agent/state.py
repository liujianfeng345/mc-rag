"""状态定义"""

from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class RAGState(TypedDict):
    """
    RAG 图的状态。

    messages:    对话历史（自动合并同角色消息）
    question:    当前用户问题
    documents:   检索到的文档列表
    generation:  生成的答案
    rewrite_count: 重写次数（防止无限循环）
    """

    messages: Annotated[list, add_messages]
    question: str
    documents: list[Document] | None
    generation: str | None
    rewrite_count: int
