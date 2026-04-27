"""agent_v3 - 基于 open_deep_research 思路的深度 RAG 代理

核心改进（相比 v2）：
- 问题分解：LLM 将复杂问题拆分为 2-4 个子问题
- 并行研究：每个子问题独立进行 ReAct 循环（多轮、多角度检索）
- 自主检索：研究员自主决定检索内容和检索策略
- 综合合成：汇总所有研究发现生成结构化答案
"""

from .graph import build_rag_graph

__all__ = ["build_rag_graph"]
