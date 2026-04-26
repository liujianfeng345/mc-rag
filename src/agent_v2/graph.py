"""构建图"""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from .state import RAGState
from .node import (
    retrieve_node,
    grade_node,
    generate_node,
    rewrite_node,
)
from ..vector.vector_store import VectorStore


# =============================================================================
# 图构建
# =============================================================================
def build_rag_graph(vector_store: VectorStore = None) -> StateGraph:
    """
    构建 Agentic RAG 图。

    参数：
        vector_store: 向量存储实例（如果为 None，则在调用时创建）

    返回：
        编译后的 StateGraph 实例（可执行 graph.invoke()）

    使用方式：
        store = VectorStore()
        graph = build_rag_graph(store)
        result = graph.invoke({"question": "如何自定义物品？"})
    """

    if vector_store is None:
        vector_store = VectorStore()

    # 创建图
    workflow = StateGraph(RAGState)

    # 添加节点 - 使用闭包将 vector_store 绑定到检索节点
    async def _retrieve(state: RAGState) -> dict:
        return await retrieve_node(state, vector_store)

    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    # 添加边
    workflow.add_edge(START, "retrieve")

    # 编译图
    return workflow.compile()
