"""agent_v4 - Self-Corrective RAG 代理

核心改进（相比 v3）：
- 自我反思：reflect 节点对生成的答案进行系统性质量评估
- 定向精炼：refine 节点针对反思发现的缺口进行补充检索与答案修订
- 循环修正：reflect → refine → synthesize 循环，最多 2 轮精炼

继承 v3 全部节点（decompose、research_batch、synthesize），在其基础上新增
reflect 和 refine 两个节点，实现答案质量的自我检查与迭代改进。
"""

from .graph import build_rag_graph

__all__ = ["build_rag_graph"]
