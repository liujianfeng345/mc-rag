"""状态定义"""

from typing import Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    RAG 图的状态。

    messages:    对话历史（自动合并同角色消息）
    question:    当前用户问题
    current_query: 改写后的问题
    documents:   检索到的文档列表
    generation:  生成的答案
    rewrite_count: 重写次数（防止无限循环）
    """

    messages: Annotated[list, add_messages]
    question: str
    current_query: str = ""
    documents: list[Document] | None
    generation: str | None
    rewrite_count: int = 0


class DocumentRelevance(BaseModel):
    """文档相关性评估结果"""
    # is_relevant: bool = Field(
    #     description="文档是否相关。True表示相关, False表示不相关。"
    # )
    relevant: Literal["相关", "不相关"] = Field(
        description="文档是否相关，只能为'相关'或'不相关'"
    )
