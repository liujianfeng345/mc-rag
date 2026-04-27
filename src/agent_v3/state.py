"""状态定义"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class AgentState(TypedDict):
    """深度 RAG 代理的状态

    messages:       对话历史（自动合并同角色消息）
    question:       用户原始问题
    research_plan:  分解后的子问题列表
    findings:       各子问题的研究发现（含主题、内容、来源）
    final_report:   最终生成的回答
    documents:      引用的文档列表（用于来源展示）
    """

    messages: Annotated[list, add_messages]
    question: str
    research_plan: list[str]
    findings: list[dict]
    final_report: str
    documents: list[Document]
