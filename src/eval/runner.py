"""
统一评测入口，整合检索评测与生成质量评测。

评测模式：
1. --retrieval-only          仅评测检索质量（默认需标注，加 --auto 用 LLM 自动判定）
2. --ragas-only              仅评测生成质量（LLM 裁判，无需标注）
3. 默认                      同时执行检索评测和生成质量评测
4. --auto                    检索评测使用 LLM 自动判定相关性（无需人工标注 relevant_sources）
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
    retrieval_mode: str = ""  # "human" 或 "auto"

    def print(self, console: Console | None = None) -> None:
        console = console or Console()

        mode_note = ""
        if self.retrieval_mode == "auto":
            mode_note = "\n[dim]（检索评测使用 LLM 自动标注模式）[/dim]"

        console.print(Panel(
            f"[bold]RAG 评测报告[/bold]\n"
            f"数据集: {self.dataset_name}\n"
            f"耗时: {self.elapsed_seconds:.1f}s"
            f"{mode_note}",
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
            "retrieval_mode": self.retrieval_mode,
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
    auto_retrieval: bool = False,
    k_values: list[int] = None,
    save_report_path: str = "",
) -> FullEvalReport:
    """执行评测的主入口。

    参数：
        dataset_path: 评测数据集 JSON 文件路径
        retrieval_only: 仅执行检索评测
        ragas_only: 仅执行生成质量评测
        auto_retrieval: 检索评测使用 LLM 自动判定（无需人工标注）
        k_values: 检索评测的 K 值列表
        save_report_path: 报告保存路径（JSON）
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

    # 检索评测决策逻辑：
    # - 如果用户指定 --ragas-only，跳过检索
    # - 如果数据集有标注 且 没指定 --auto，用精确评测
    # - 如果数据集无标注 或 指定 --auto，用 LLM 自动评测
    # - 如果数据集无标注 且 没指定 --auto 且 没指定 --retrieval-only，跳过检索（默认行为）
    do_retrieval = not ragas_only
    if retrieval_only and not dataset.has_relevance_labels and not auto_retrieval:
        console.print(
            "[yellow]⚠ 数据集没有 relevant_sources 标注，且未指定 --auto 模式[/yellow]\n"
            "[yellow]   检索评测将被跳过。如需自动评测，请使用: --retrieval-only --auto[/yellow]\n"
        )
        do_retrieval = False

    use_auto = auto_retrieval or (do_retrieval and not dataset.has_relevance_labels)

    do_ragas = not retrieval_only or (retrieval_only and not do_retrieval and not use_auto)
    # 修正：如果只做检索，不做 ragas
    if retrieval_only and do_retrieval:
        do_ragas = False

    report = FullEvalReport(dataset_name=dataset.name)

    # 检索评测
    if do_retrieval:
        retrieval_eval = RetrievalEvaluator(store)
        if use_auto:
            console.print("[bold cyan]━━━ 检索质量评测（LLM 自动标注模式）━━━[/bold cyan]")
            console.print("[dim]使用 LLM 自动判断每个检索结果与问题的相关性，无需人工标注[/dim]")
            report.retrieval_mode = "auto"
            report.retrieval_result = await retrieval_eval.evaluate_auto(
                dataset, k_values=k_values
            )
        else:
            console.print("[bold cyan]━━━ 检索质量评测（人工标注模式）━━━[/bold cyan]")
            report.retrieval_mode = "human"
            report.retrieval_result = await retrieval_eval.evaluate(
                dataset, k_values=k_values
            )

    # 生成质量评测
    if do_ragas:
        if not do_retrieval:
            console.print()
        console.print("[bold cyan]━━━ 生成质量评测（LLM 裁判）━━━[/bold cyan]")
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
