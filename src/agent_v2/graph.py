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
# 路由决策
# =============================================================================
def grade_router(state: RAGState) -> Literal["generate", "rewrite"]:
    """
    评分后的路由决策。

    - 如果已有足够的相关文档 → 进入生成
    - 如果重写超过2次仍无相关文档 → 强制生成（给出降级回答）
    - 否则 → 进入重写，优化查询
    """
    documents = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    # 有文档且未超过重写限制 → 直接生成
    if len(documents) >= 2:
        return "generate"

    # 超过重写限制 → 强制生成（哪怕文档质量不高）
    if rewrite_count >= 2:
        return "generate"

    # 文档不足 → 重写查询
    return "rewrite"

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
    workflow.add_edge("retrieve", "grade")

    # 条件路由：评分结果决定下一步
    workflow.add_conditional_edges(
        "grade",
        grade_router,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # 重写后回到检索
    workflow.add_edge("rewrite", "retrieve")

    # 生成后结束
    workflow.add_edge("generate", END)

    # 编译图
    return workflow.compile()
