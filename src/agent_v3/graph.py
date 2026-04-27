"""构建深度 RAG 图

图结构（线性，无循环）：
    START → decompose → research_batch → synthesize → END

与 v2 的区别：
    - 无显式的 grade/rewrite 循环，改为研究员的 ReAct 自主迭代
    - 多子问题并行研究，每个研究员独立进行多轮检索
    - 综合合成阶段统一汇总所有研究发现
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .node import (
    decompose_node,
    research_batch_node,
    synthesize_node,
)
from ..vector.vector_store import VectorStore


def build_rag_graph(vector_store: VectorStore = None) -> StateGraph:
    """
    构建深度 RAG 图。

    参数：
        vector_store: 向量存储实例（为 None 时自动创建）

    返回：
        编译后的 StateGraph 实例

    使用方式：
        store = VectorStore()
        graph = build_rag_graph(store)
        result = await graph.ainvoke({"question": "如何自定义物品的属性？"})
    """
    if vector_store is None:
        vector_store = VectorStore()

    workflow = StateGraph(AgentState)

    # 将 vector_store 注入到研究节点（闭包方式）
    async def _research_batch(state: AgentState) -> dict:
        return await research_batch_node(state, vector_store)

    workflow.add_node("decompose", decompose_node)
    workflow.add_node("research_batch", _research_batch)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "research_batch")
    workflow.add_edge("research_batch", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()
