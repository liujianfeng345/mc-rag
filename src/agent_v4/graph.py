"""构建 v4 Self-Corrective RAG 图

图结构（带自我反思与精炼循环）：
    START → decompose → research_batch → synthesize → reflect
                                                         │
                                    ┌────────────────────┼──────────────────┐
                                    │ 答案充分            │                  │ 有缺口 且 refine_count < 2
                                    ▼                    │                  ▼
                                  END                    │              refine
                                                         │                  │
                                    （refine_count >= 2）│                  │
                                                         │                  ▼
                                                         │            synthesize（回到 synthesize）
                                                         │
与 v3 的区别：
    - 新增 reflect 节点：对答案质量进行系统性评估
    - 新增 refine 节点：针对反思发现的缺口进行定向补充检索
    - 条件路由：根据反思结果决定继续精炼或结束
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .node import (
    decompose_node,
    research_batch_node,
    synthesize_node,
    reflect_node,
    refine_node,
)
from ..vector.vector_store import VectorStore
from ..utils.config import MAX_REFINE_ITERATIONS


def _route_after_reflect(state: AgentState) -> str:
    """根据反思结果决定下一步。"""
    reflection = state.get("reflection", {})
    refine_count = state.get("refine_count", 0)

    if reflection.get("is_sufficient", True):
        return END
    if refine_count < MAX_REFINE_ITERATIONS:
        return "refine"
    return END  # 达到上限，降级退出


def build_rag_graph(vector_store: VectorStore = None) -> StateGraph:
    """
    构建 v4 Self-Corrective RAG 图。

    参数：
        vector_store: 向量存储实例（为 None 时自动创建）

    返回：
        编译后的 StateGraph 实例
    """
    if vector_store is None:
        vector_store = VectorStore()

    workflow = StateGraph(AgentState)

    # 将 vector_store 注入到需要检索的节点
    async def _research_batch(state: AgentState) -> dict:
        return await research_batch_node(state, vector_store)

    async def _refine(state: AgentState) -> dict:
        return await refine_node(state, vector_store)

    workflow.add_node("decompose", decompose_node)
    workflow.add_node("research_batch", _research_batch)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("refine", _refine)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "research_batch")
    workflow.add_edge("research_batch", "synthesize")
    workflow.add_edge("synthesize", "reflect")
    workflow.add_conditional_edges("reflect", _route_after_reflect, {
        "refine": "refine",
        END: END,
    })
    workflow.add_edge("refine", "synthesize")

    return workflow.compile()
