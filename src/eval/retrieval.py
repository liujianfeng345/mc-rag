"""
检索质量评测。

支持的指标：
- Recall@K：前 K 个结果中召回了多少相关文档（关心"找全了没"）
- Precision@K：前 K 个结果中有多少是真正相关的（关心"找对了没"）
- MRR (Mean Reciprocal Rank)：第一个相关文档的平均排名倒数（关心"首个命中靠不靠前"）
- NDCG@K：带位置权重的排序质量（关心"排序对不对"）
- Hit@K：前 K 个结果中是否至少命中一个相关文档

两种评测模式：
1. evaluate()：使用人工标注 relevant_sources，精确但需手动标注
2. evaluate_auto()：使用 LLM 自动判断检索结果相关性，无需标注，适合快速对比

使用方式：
    from src.eval.retrieval import RetrievalEvaluator
    from src.vector.vector_store import VectorStore

    store = VectorStore()
    evaluator = RetrievalEvaluator(store)

    # 精确评测（需标注）
    result = await evaluator.evaluate(dataset, k=5)

    # 自动评测（无需标注，LLM 裁判）
    result = await evaluator.evaluate_auto(dataset, k=5)
"""

import asyncio
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .dataset import EvalDataset, EvalItem
from ..vector.vector_store import VectorStore
from ..utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL


# 检索结果相关性判断 prompt
RELEVANCE_JUDGE_PROMPT = """你的任务是判断一个"检索到的文档片段"是否与用户问题相关，能否帮助回答这个问题。

判断标准：
- 相关（YES）：文档内容包含能回答问题的信息，或者与问题主题直接相关
- 不相关（NO）：文档内容与问题无关，或者只提到了关键词但不涉及实质性内容

用户问题：{question}

检索到的文档片段（来源: {source}）：
{content}

请回答 YES 或 NO，并简要说明理由。用 JSON 格式输出：
{{"verdict": "YES/NO", "reason": "简短理由"}}"""


@dataclass
class RetrievalMetrics:
    """单次检索的指标。"""
    question: str
    recall_at_k: dict[int, float] = field(default_factory=dict)    # K → Recall
    precision_at_k: dict[int, float] = field(default_factory=dict) # K → Precision
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)      # K → NDCG
    hit_at_k: dict[int, bool] = field(default_factory=dict)        # K → Hit
    retrieved_sources: list[str] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """完整评测结果。"""
    dataset_name: str
    metrics: list[RetrievalMetrics]
    k_values: list[int]

    @property
    def avg_recall(self) -> dict[int, float]:
        """平均 Recall@K。"""
        avg = defaultdict(float)
        for m in self.metrics:
            for k, v in m.recall_at_k.items():
                avg[k] += v
        n = max(len(self.metrics), 1)
        return {k: v / n for k, v in avg.items()}

    @property
    def avg_precision(self) -> dict[int, float]:
        """平均 Precision@K。"""
        avg = defaultdict(float)
        for m in self.metrics:
            for k, v in m.precision_at_k.items():
                avg[k] += v
        n = max(len(self.metrics), 1)
        return {k: v / n for k, v in avg.items()}

    @property
    def avg_mrr(self) -> float:
        return sum(m.mrr for m in self.metrics) / max(len(self.metrics), 1)

    @property
    def avg_ndcg(self) -> dict[int, float]:
        avg = defaultdict(float)
        for m in self.metrics:
            for k, v in m.ndcg_at_k.items():
                avg[k] += v
        n = max(len(self.metrics), 1)
        return {k: v / n for k, v in avg.items()}

    @property
    def avg_hit(self) -> dict[int, float]:
        avg = defaultdict(float)
        for m in self.metrics:
            for k, v in m.hit_at_k.items():
                avg[k] += 1.0 if v else 0.0
        n = max(len(self.metrics), 1)
        return {k: v / n for k, v in avg.items()}

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [f"数据集: {self.dataset_name} | 问题数: {len(self.metrics)}"]
        for k in self.k_values:
            lines.append(
                f"  K={k}: Recall={self.avg_recall[k]:.3f}  "
                f"Precision={self.avg_precision[k]:.3f}  "
                f"NDCG={self.avg_ndcg[k]:.3f}  "
                f"Hit@{k}={self.avg_hit[k]:.3f}"
            )
        lines.append(f"  MRR={self.avg_mrr:.3f}")
        return "\n".join(lines)

    def rich_table(self) -> Table:
        """生成 Rich 表格。"""
        table = Table(title=f"检索评测: {self.dataset_name}")
        table.add_column("K", justify="center", style="cyan")
        table.add_column("Recall", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("NDCG", justify="right")
        table.add_column("Hit Rate", justify="right")

        for k in self.k_values:
            table.add_row(
                str(k),
                f"{self.avg_recall[k]:.1%}",
                f"{self.avg_precision[k]:.1%}",
                f"{self.avg_ndcg[k]:.3f}",
                f"{self.avg_hit[k]:.1%}",
            )
        table.add_row("─" * 4, "─" * 7, "─" * 10, "─" * 9, "─" * 9)
        table.add_row(
            "MRR",
            f"{self.avg_mrr:.3f}",
            "(首个相关文档的倒数排名平均值)",
            "",
            "",
        )
        return table


class RetrievalEvaluator:
    """检索器评测器。

    对每条评测数据执行 hybrid_search，计算检索质量指标。

    使用方式：
        store = VectorStore()
        evaluator = RetrievalEvaluator(store)
        result = evaluator.evaluate(dataset, k_values=[3, 5, 10])
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    async def evaluate(
        self,
        dataset: EvalDataset,
        k_values: list[int] = None,
        verbose: bool = True,
    ) -> EvalResult:
        """对数据集执行检索评测。"""
        if not dataset.has_relevance_labels:
            raise ValueError("数据集中没有相关文档标注（relevant_sources），无法做检索评测")

        k_values = k_values or [3, 5, 10]
        console = Console() if verbose else None
        all_metrics: list[RetrievalMetrics] = []

        for idx, item in enumerate(dataset.items):
            # 执行混合检索
            docs = await self.store.hybrid_search(item.question, top_k=max(k_values))

            retrieved_sources = []
            for doc in docs:
                source = doc.metadata.get("source", "未知")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                retrieved_sources.append(f"{source}#{chunk_idx}")

            metrics = self._compute_metrics(item, retrieved_sources, k_values)
            all_metrics.append(metrics)

            if console:
                self._print_item_progress(console, idx + 1, len(dataset), item.question, metrics)

        result = EvalResult(
            dataset_name=dataset.name,
            metrics=all_metrics,
            k_values=k_values,
        )

        if console:
            console.print(result.rich_table())

        return result

    def _compute_metrics(
        self,
        item: EvalItem,
        retrieved_sources: list[str],
        k_values: list[int],
    ) -> RetrievalMetrics:
        """计算单条评测的全套检索指标。"""
        relevant_set = set(item.relevant_sources)

        # 辅助函数：判断第 i 个结果是否相关
        def _is_relevant(source_str: str) -> bool:
            # 归一化路径分隔符（ChromaDB 存 \，数据集用 /）
            base = source_str.rsplit("#", 1)[0] if "#" in source_str else source_str
            base = base.replace("\\", "/")
            for rel in relevant_set:
                rel_norm = rel.replace("\\", "/")
                if base == rel_norm or base.startswith(rel_norm) or rel_norm.startswith(base):
                    return True
            return False

        # ----- Recall@K & Precision@K & Hit@K -----
        recall_at_k: dict[int, float] = {}
        precision_at_k: dict[int, float] = {}
        hit_at_k: dict[int, bool] = {}

        for k in k_values:
            top_k = retrieved_sources[:k]
            hits = sum(1 for s in top_k if _is_relevant(s))
            recall_at_k[k] = hits / len(relevant_set) if relevant_set else 0.0
            precision_at_k[k] = hits / k
            hit_at_k[k] = hits > 0

        # ----- MRR -----
        mrr = 0.0
        for rank, source in enumerate(retrieved_sources):
            if _is_relevant(source):
                mrr = 1.0 / (rank + 1)
                break

        # ----- NDCG@K -----
        ndcg_at_k: dict[int, float] = {}
        for k in k_values:
            top_k = retrieved_sources[:k]
            dcg = 0.0
            for i, source in enumerate(top_k):
                rel = 1.0 if _is_relevant(source) else 0.0
                dcg += rel / math.log2(i + 2)  # i+2 因为 log2(1)=0

            # IDCG：理想情况下前 min(k, |relevant|) 个都是相关的
            idcg = 0.0
            ideal_hits = min(k, len(relevant_set))
            for i in range(ideal_hits):
                idcg += 1.0 / math.log2(i + 2)

            ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0

        return RetrievalMetrics(
            question=item.question,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_at_k=hit_at_k,
            retrieved_sources=retrieved_sources[:max(k_values)],
            relevant_sources=list(relevant_set),
        )

    async def evaluate_auto(
        self,
        dataset: EvalDataset,
        k_values: list[int] = None,
        verbose: bool = True,
    ) -> EvalResult:
        """
        无标注检索评测 —— 用 LLM 自动判断每个检索结果是否与问题相关。

        数据集只需要 question 字段，不依赖 relevant_sources 标注。
        LLM 裁判的准确性不如人工标注，但用于横向对比（如 v1 vs v4）足够可靠。

        使用方式：
            dataset = EvalDataset.from_json("my_questions.json")  # 只需要 question
            result = await evaluator.evaluate_auto(dataset)
        """
        k_values = k_values or [3, 5, 10]
        console = Console() if verbose else None

        # 评测专用 LLM（temperature=0 保证裁判一致性）
        judge_llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
            max_tokens=512,
        )

        all_metrics: list[RetrievalMetrics] = []

        for idx, item in enumerate(dataset.items):
            if console:
                console.print(f"[dim][{idx + 1}/{len(dataset)}] 检索并评测: {item.question[:80]}...[/dim]")

            # 执行混合检索
            docs = await self.store.hybrid_search(item.question, top_k=max(k_values))

            retrieved_sources = []
            judge_tasks = []

            for doc in docs:
                source = doc.metadata.get("source", "未知")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                source_id = f"{source}#{chunk_idx}"
                retrieved_sources.append(source_id)

                # 构造 LLM 裁判请求
                judge_tasks.append(
                    judge_llm.ainvoke([
                        HumanMessage(content=RELEVANCE_JUDGE_PROMPT.format(
                            question=item.question,
                            source=source,
                            content=doc.page_content[:2000],  # 截断长文档
                        ))
                    ])
                )

            # 并行判断所有检索结果的相关性
            judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

            # 收集 LLM 判定为相关的文档源
            auto_relevant = []
            for i, result in enumerate(judge_results):
                if isinstance(result, Exception):
                    continue
                try:
                    verdict = self._parse_judge_verdict(result.content)
                    if verdict == "YES":
                        auto_relevant.append(retrieved_sources[i])
                except Exception:
                    continue

            # 用 LLM 判定结果作为弱标注计算指标
            metrics = self._compute_metrics_auto(
                question=item.question,
                retrieved_sources=retrieved_sources,
                auto_relevant=auto_relevant,
                k_values=k_values,
            )
            all_metrics.append(metrics)

            if console:
                self._print_item_progress(console, idx + 1, len(dataset), item.question, metrics)

        result = EvalResult(
            dataset_name=dataset.name,
            metrics=all_metrics,
            k_values=k_values,
        )

        if console:
            console.print()
            console.print("[dim]⚠ LLM 自动标注模式：相关性由 LLM 判定，指标供横向对比参考，非绝对精度[/dim]")
            console.print(result.rich_table())

        return result

    def _compute_metrics_auto(
        self,
        question: str,
        retrieved_sources: list[str],
        auto_relevant: list[str],
        k_values: list[int],
    ) -> RetrievalMetrics:
        """计算基于 LLM 自动标注的检索指标，逻辑与 _compute_metrics 一致。"""
        relevant_set = set(auto_relevant)

        def _is_relevant(source_str: str) -> bool:
            # 归一化路径分隔符
            return source_str.replace("\\", "/") in relevant_set

        # Recall@K & Precision@K & Hit@K
        recall_at_k: dict[int, float] = {}
        precision_at_k: dict[int, float] = {}
        hit_at_k: dict[int, bool] = {}

        for k in k_values:
            top_k = retrieved_sources[:k]
            hits = sum(1 for s in top_k if _is_relevant(s))
            recall_at_k[k] = hits / len(relevant_set) if relevant_set else 0.0
            precision_at_k[k] = hits / k
            hit_at_k[k] = hits > 0

        # MRR
        mrr = 0.0
        for rank, source in enumerate(retrieved_sources):
            if _is_relevant(source):
                mrr = 1.0 / (rank + 1)
                break

        # NDCG@K
        ndcg_at_k: dict[int, float] = {}
        for k in k_values:
            top_k = retrieved_sources[:k]
            dcg = 0.0
            for i, source in enumerate(top_k):
                rel = 1.0 if _is_relevant(source) else 0.0
                dcg += rel / math.log2(i + 2)

            idcg = 0.0
            ideal_hits = min(k, len(relevant_set))
            for i in range(ideal_hits):
                idcg += 1.0 / math.log2(i + 2)

            ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0

        return RetrievalMetrics(
            question=question,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_at_k=hit_at_k,
            retrieved_sources=retrieved_sources[:max(k_values)],
            relevant_sources=list(relevant_set),
        )

    @staticmethod
    def _parse_judge_verdict(text: str) -> str:
        """从 LLM 输出中解析相关性判定结果。"""
        if not text:
            return "NO"
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        try:
            result = json.loads(text)
            return result.get("verdict", "NO") if isinstance(result, dict) else "NO"
        except json.JSONDecodeError:
            # 退化情况：直接在文本中搜索 YES/NO
            if "YES" in text.upper():
                return "YES"
            return "NO"

    def _print_item_progress(
        self,
        console: Console,
        idx: int,
        total: int,
        question: str,
        metrics: RetrievalMetrics,
    ) -> None:
        hit_status = "✓" if metrics.hit_at_k.get(5, False) else "✗"
        color = "green" if metrics.hit_at_k.get(5, False) else "red"
        mrr_str = f"MRR={metrics.mrr:.3f}"
        console.print(
            f"  [{color}]{hit_status}[/{color}] [{idx}/{total}] {question[:60]}... "
            f"[dim]({mrr_str})[/dim]"
        )
