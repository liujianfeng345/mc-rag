"""
检索质量评测。

支持的指标：
- Recall@K：前 K 个结果中召回了多少相关文档（关心"找全了没"）
- Precision@K：前 K 个结果中有多少是真正相关的（关心"找对了没"）
- MRR (Mean Reciprocal Rank)：第一个相关文档的平均排名倒数（关心"首个命中靠不靠前"）
- NDCG@K：带位置权重的排序质量（关心"排序对不对"）
- Hit@K：前 K 个结果中是否至少命中一个相关文档

使用方式：
    from src.eval.retrieval import RetrievalEvaluator
    from src.vector.vector_store import VectorStore

    store = VectorStore()
    evaluator = RetrievalEvaluator(store)
    result = await evaluator.evaluate(dataset, k=5)
    print(result.summary())
"""

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .dataset import EvalDataset, EvalItem
from ..vector.vector_store import VectorStore


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
            # 去掉 chunk 索引后缀做模糊匹配
            base = source_str.rsplit("#", 1)[0] if "#" in source_str else source_str
            for rel in relevant_set:
                if base == rel or base.startswith(rel) or rel.startswith(base):
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
