# src/benchmark/profiler.py
"""性能打点 — 端到端耗时、阶段拆解、首 token 延迟。"""

import time
from dataclasses import dataclass, field


@dataclass
class QuestionTiming:
    """单条问题的耗时明细（毫秒）。"""
    question: str = ""
    retrieval_ms: float = 0.0
    ttft_ms: float = 0.0       # 首 token 到达时间
    generation_ms: float = 0.0  # 首 token → 末 token
    total_ms: float = 0.0


@dataclass
class VersionTiming:
    """单个版本的耗时汇总。"""
    agent_version: str = ""
    per_question: list[QuestionTiming] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        vals = [t.ttft_ms for t in self.per_question if t.ttft_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_retrieval_ms(self) -> float:
        vals = [t.retrieval_ms for t in self.per_question if t.retrieval_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_generation_ms(self) -> float:
        vals = [t.generation_ms for t in self.per_question if t.generation_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_total_ms(self) -> float:
        vals = [t.total_ms for t in self.per_question]
        return sum(vals) / len(vals) if vals else 0.0


async def run_with_timing(
    graph,
    input_state: dict,
    output_node: str,
) -> tuple[dict, QuestionTiming]:
    """执行一次 agent graph 调用并打点。

    通过 astream_events 捕获流式事件：
    - 第一个 on_chat_model_stream 事件 → 记录首 token 时间
    - 最后一个 on_chat_model_stream 事件 → 记录生成结束时间
    - on_chain_end → 收集最终 state

    返回：
        (final_state, timing_info)
    """
    timing = QuestionTiming()
    t_start = time.perf_counter()

    final_state: dict = {}
    first_token_seen = False
    t_first_token = t_start
    t_last_token = t_start

    async for event in graph.astream_events(input_state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node == output_node:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    if not first_token_seen:
                        t_first_token = time.perf_counter()
                        first_token_seen = True
                    t_last_token = time.perf_counter()

        if kind == "on_chain_end" and isinstance(
            event.get("data", {}).get("output"), dict
        ):
            final_state.update(event["data"]["output"])

    t_end = time.perf_counter()

    timing.total_ms = (t_end - t_start) * 1000
    if first_token_seen:
        timing.ttft_ms = (t_first_token - t_start) * 1000
        timing.generation_ms = (t_last_token - t_first_token) * 1000

    return final_state, timing


async def measure_retrieval(
    store,
    question: str,
    top_k: int = 5,
) -> tuple[list, float]:
    """单独测量一次检索耗时，返回 (文档列表, 毫秒)。"""
    t0 = time.perf_counter()
    docs = await store.hybrid_search(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return docs, elapsed_ms
