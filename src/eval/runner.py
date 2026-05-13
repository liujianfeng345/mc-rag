"""
统一评测入口，整合检索评测与生成质量评测。

提供三种评测模式：
1. --retrieval-only  仅评测检索质量（需要 relevant_sources 标注）
2. --ragas-only      仅评测生成质量（使用 LLM 裁判）
3. 默认              同时执行检索评测和生成质量评测
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .dataset import EvalDataset
from .retrieval import RetrievalEvaluator, EvalResult as RetrievalResult
from .ragas_eval import RAGASEvaluator, RAGASResult
from ..vector.vector_store import VectorStore


@dataclass
class FullEvalReport:
    """完整评测报告。"""
    dataset_name: str
    retrieval_result: RetrievalResult | None = None
    ragas_result: RAGASResult | None = None
    elapsed_seconds: float = 0.0

    def print(self, console: Console | None = None) -> None:
        console = console or Console()

        console.print(Panel(
            f"[bold]RAG 评测报告[/bold]\n"
            f"数据集: {self.dataset_name}\n"
            f"耗时: {self.elapsed_seconds:.1f}s",
            border_style="cyan",
        ))

        if self.retrieval_result:
            console.print(self.retrieval_result.rich_table())

        if self.ragas_result:
            console.print(self.ragas_result.rich_table())

    def to_dict(self) -> dict:
        """导出为字典，方便保存为 JSON。"""
        report = {
            "dataset_name": self.dataset_name,
            "elapsed_seconds": self.elapsed_seconds,
        }

        if self.retrieval_result:
            report["retrieval"] = {
                "avg_recall": self.retrieval_result.avg_recall,
                "avg_precision": self.retrieval_result.avg_precision,
                "avg_ndcg": self.retrieval_result.avg_ndcg,
                "avg_hit": self.retrieval_result.avg_hit,
                "avg_mrr": self.retrieval_result.avg_mrr,
                "per_question": [
                    {
                        "question": m.question,
                        "recall_at_5": m.recall_at_k.get(5, 0),
                        "precision_at_5": m.precision_at_k.get(5, 0),
                        "mrr": m.mrr,
                    }
                    for m in self.retrieval_result.metrics
                ],
            }

        if self.ragas_result:
            report["ragas"] = {
                "avg_faithfulness": self.ragas_result.avg_faithfulness,
                "avg_answer_relevance": self.ragas_result.avg_answer_relevance,
                "avg_context_relevance": self.ragas_result.avg_context_relevance,
                "per_question": [
                    {
                        "question": m.question,
                        "faithfulness": m.faithfulness,
                        "answer_relevance": m.answer_relevance,
                        "context_relevance": m.context_relevance,
                        "answer": m.answer[:500] if m.answer else "",
                        "error": m.error,
                    }
                    for m in self.ragas_result.metrics
                ],
            }

        return report


async def run_eval(
    dataset_path: str,
    retrieval_only: bool = False,
    ragas_only: bool = False,
    k_values: list[int] = None,
    save_report_path: str = "",
) -> FullEvalReport:
    """执行评测的主入口。

    参数：
        dataset_path: 评测数据集 JSON 文件路径
        retrieval_only: 仅执行检索评测
        ragas_only: 仅执行生成质量评测
        k_values: 检索评测的 K 值列表
        save_report_path: 报告保存路径（JSON）

    返回：
        FullEvalReport: 完整评测报告
    """
    console = Console()
    start_time = time.time()

    # 加载数据集
    console.print(f"[bold]加载评测数据集: {dataset_path}[/bold]")
    dataset = EvalDataset.from_json(dataset_path)
    stats = dataset.stats()
    console.print(f"  问题数: {stats['问题数量']}")
    console.print(f"  有相关文档标注: {stats['有相关文档标注']}")
    console.print(f"  有参考答案: {stats['有参考答案']}")

    store = VectorStore()
    store_stats = await store.stats()
    console.print(f"  知识库文档块: {store_stats['文档块数量']}\n")

    report = FullEvalReport(dataset_name=dataset.name)

    do_retrieval = not ragas_only and dataset.has_relevance_labels
    do_ragas = not retrieval_only

    # 检索评测
    if do_retrieval:
        console.print("[bold cyan]━━━ 检索质量评测 ━━━[/bold cyan]")
        retrieval_eval = RetrievalEvaluator(store)
        report.retrieval_result = await retrieval_eval.evaluate(
            dataset, k_values=k_values
        )

    # 生成质量评测
    if do_ragas:
        console.print("\n[bold cyan]━━━ 生成质量评测（LLM 裁判）━━━[/bold cyan]")
        console.print("[dim]提示：评测过程需要调用 LLM，数据集越大耗时越长[/dim]")
        ragas_eval = RAGASEvaluator(store)
        report.ragas_result = await ragas_eval.evaluate(dataset)

    report.elapsed_seconds = time.time() - start_time

    # 打印综合报告
    report.print(console)

    # 保存报告
    if save_report_path:
        save_path = Path(save_report_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[green]评测报告已保存到: {save_report_path}[/green]")

    return report
