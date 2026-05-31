# src/benchmark/runner.py
"""Benchmark 核心调度 — 多版本循环、调用 agent 图、汇总结果。"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..eval.dataset import EvalDataset
from ..eval.retrieval import RetrievalEvaluator, EvalResult as RetrievalResult
from ..vector.vector_store import VectorStore
from ..utils.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, RETRIEVAL_TOP_K,
)

from .benchmark_config import DEFAULT_VERSIONS, RETRIEVAL_K_VALUES
from .database import BenchmarkDB
from .profiler import (
    QuestionTiming, VersionTiming, run_with_timing, measure_retrieval,
)

# 各版本最终答案节点名
OUTPUT_NODES = {"v4": "synthesize", "v3": "synthesize", "v2": "generate", "v1": "generate"}


def _build_graph_for_version(version: str, store: VectorStore):
    """按版本字符串动态导入并构建 agent 图。"""
    if version == "v4":
        from ..agent_v4.graph import build_rag_graph
    elif version == "v3":
        from ..agent_v3.graph import build_rag_graph
    elif version == "v2":
        from ..agent_v2.graph import build_rag_graph
    else:
        from ..agent.graph import build_rag_graph
    return build_rag_graph(store)


def _get_git_commit() -> str:
    """获取当前 HEAD commit hash。"""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


@dataclass
class VersionResult:
    """单个版本的 benchmark 结果。"""
    agent_version: str
    timing: VersionTiming
    answers: list[str] = field(default_factory=list)
    ragas_faithfulness: float = 0.0
    ragas_answer_relevance: float = 0.0
    ragas_context_relevance: float = 0.0


@dataclass
class BenchmarkReport:
    """完整 benchmark 报告。"""
    run_id: str
    dataset_name: str
    question_count: int
    git_commit: str
    retrieval: RetrievalResult | None = None
    versions: list[VersionResult] = field(default_factory=list)
    passed_map: dict[str, bool] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0


class BenchmarkRunner:
    """Benchmark 调度器。"""

    def __init__(
        self,
        dataset_path: str,
        versions: list[str] | None = None,
        profile_only: bool = False,
        eval_only: bool = False,
    ):
        self.dataset_path = dataset_path
        self.versions = versions or DEFAULT_VERSIONS
        self.profile_only = profile_only
        self.eval_only = eval_only
        self.db = BenchmarkDB()

    async def run(self) -> BenchmarkReport:
        """执行完整 benchmark 流程。"""
        t_total_start = time.perf_counter()

        dataset = EvalDataset.from_json(self.dataset_path)
        store = VectorStore()
        db = self.db
        db.init_default_baselines()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        git_commit = _get_git_commit()

        report = BenchmarkReport(
            run_id=run_id,
            dataset_name=dataset.name,
            question_count=len(dataset),
            git_commit=git_commit,
        )

        # ---- 检索评测（版本无关，跑一次） ----
        if not self.profile_only:
            retrieval_eval = RetrievalEvaluator(store)
            report.retrieval = await retrieval_eval.evaluate(
                dataset, k_values=RETRIEVAL_K_VALUES, verbose=False,
            )

        # ---- 逐版本跑 ----
        for version in self.versions:
            vresult = await self._run_version(version, dataset, store)
            report.versions.append(vresult)

        # ---- 基线对比 ----
        from .comparator import compare_all
        report.passed_map = compare_all(db, report.retrieval, report.versions, report.dataset_name)

        # ---- 写入数据库 ----
        self._persist_runs(db, report)

        report.total_elapsed_ms = (time.perf_counter() - t_total_start) * 1000
        return report

    async def _run_version(
        self, version: str, dataset, store: VectorStore,
    ) -> VersionResult:
        """对单个版本跑全部问题。"""
        from rich.console import Console
        console = Console()

        graph = _build_graph_for_version(version, store)
        output_node = OUTPUT_NODES.get(version, "generate")
        total = len(dataset)

        console.print(
            f"\n[bold cyan]{version.upper()}[/bold cyan] 开始评测 "
            f"（共 [bold]{total}[/bold] 题）"
        )

        timings: list[QuestionTiming] = []
        answers: list[str] = []

        for idx, item in enumerate(dataset.items, 1):
            # 构建输入 state
            input_state: dict = {"question": item.question}
            if version in ("v1", "v2"):
                input_state["rewrite_count"] = 0

            # 进度提示
            question_preview = item.question[:60] + "..." if len(item.question) > 60 else item.question
            remaining = total - idx
            console.print(
                f"  [dim][{idx}/{total}][/dim] {question_preview} "
                f"[dim](剩余 {remaining})[/dim]",
                end="\r",
            )

            t0 = time.perf_counter()

            if self.eval_only:
                # eval_only 模式：直接用 graph.ainvoke（不打点）
                result = await graph.ainvoke(input_state)
                answer = result.get("final_report") or result.get("generation", "")
                timing = QuestionTiming(question=item.question)
            else:
                # 正常模式：打点运行
                final_state, timing = await run_with_timing(graph, input_state, output_node)
                answer = final_state.get("final_report") or final_state.get("generation", "")
                timing.question = item.question

            elapsed = (time.perf_counter() - t0) * 1000
            console.print(
                f"  [dim][{idx}/{total}][/dim] {question_preview} "
                f"[green]✓ {elapsed:.0f}ms[/green]"
            )

            timings.append(timing)
            answers.append(answer)

        avg_total = sum(t.total_ms for t in timings) / len(timings) if timings else 0
        console.print(
            f"  [bold cyan]{version.upper()}[/bold cyan] 完成，"
            f"平均耗时 [bold]{avg_total:.0f}ms[/bold]\n"
        )

        vtiming = VersionTiming(agent_version=version, per_question=timings)

        # RAGAS 生成质量评测
        faithfulness = 0.0
        answer_relevance = 0.0
        context_relevance = 0.0
        if not self.profile_only:
            console.print(
                f"  [dim]{version.upper()} RAGAS 生成质量评测中...[/dim]"
            )
            ragas_metrics = await _compute_ragas_batch(dataset, answers, store)
            faithfulness = ragas_metrics["avg_faithfulness"]
            answer_relevance = ragas_metrics["avg_answer_relevance"]
            context_relevance = ragas_metrics["avg_context_relevance"]

        return VersionResult(
            agent_version=version,
            timing=vtiming,
            answers=answers,
            ragas_faithfulness=faithfulness,
            ragas_answer_relevance=answer_relevance,
            ragas_context_relevance=context_relevance,
        )

    def _persist_runs(self, db: BenchmarkDB, report: BenchmarkReport) -> None:
        """将报告写入 SQLite。"""
        retrieval = report.retrieval
        for vr in report.versions:
            run_data = {
                "id": f"{report.run_id}-{vr.agent_version}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_version": vr.agent_version,
                "dataset_name": report.dataset_name,
                "question_count": report.question_count,
                "git_commit": report.git_commit,
                "total_duration_ms": vr.timing.avg_total_ms,
                "avg_ttft_ms": vr.timing.avg_ttft_ms,
                "avg_retrieval_ms": vr.timing.avg_retrieval_ms,
                "avg_generation_ms": vr.timing.avg_generation_ms,
                "recall_at_5": retrieval.avg_recall.get(5, 0) if retrieval else 0,
                "precision_at_5": retrieval.avg_precision.get(5, 0) if retrieval else 0,
                "mrr": retrieval.avg_mrr if retrieval else 0,
                "ndcg_at_5": retrieval.avg_ndcg.get(5, 0) if retrieval else 0,
                "hit_at_5": retrieval.avg_hit.get(5, 0) if retrieval else 0,
                "faithfulness": vr.ragas_faithfulness,
                "answer_relevance": vr.ragas_answer_relevance,
                "context_relevance": vr.ragas_context_relevance,
                "passed": 1 if report.passed_map.get(vr.agent_version, False) else 0,
                "report_json": json.dumps(_build_report_dict(report, vr), ensure_ascii=False),
            }
            db.insert_run(run_data)


# ---- RAGAS 评测逻辑（内联，复用 ragas_eval.py 的 prompt 和计算方式） ----

from ..eval.ragas_eval import (
    FAITHFULNESS_STATEMENTS_PROMPT,
    FAITHFULNESS_VERIFY_PROMPT,
    ANSWER_RELEVANCE_REVERSE_PROMPT,
    CONTEXT_RELEVANCE_EXTRACT_PROMPT,
)

import numpy as np
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


async def _compute_ragas_batch(dataset, answers, store) -> dict:
    """批量计算 RAGAS 指标（Faithfulness, Answer Relevance, Context Relevance）。"""
    llm = ChatOpenAI(
        model=LLM_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        temperature=0.0, max_tokens=2048,
    )
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    all_faith = []
    all_answer_rel = []
    all_context_rel = []

    for idx, item in enumerate(dataset.items):
        question = item.question
        answer = answers[idx]

        docs = await store.hybrid_search(question, top_k=RETRIEVAL_TOP_K)
        context = "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}" for d in docs
        )

        # Faithfulness
        statements = await _extract_statements(llm, answer)
        if statements:
            verdicts = await asyncio.gather(*[
                _verify_statement(llm, s, context) for s in statements
            ])
            yes_count = sum(1 for v in verdicts if v == "YES")
            all_faith.append(yes_count / len(statements))
        else:
            all_faith.append(1.0)

        # Answer Relevance
        reversed_qs = await _reverse_questions(llm, answer)
        if reversed_qs:
            all_texts = [question] + reversed_qs
            all_embeds = await asyncio.to_thread(embeddings.embed_documents, all_texts)
            q_embed = np.array(all_embeds[0])
            gen_embeds = np.array(all_embeds[1:])
            q_norm = q_embed / (np.linalg.norm(q_embed) + 1e-8)
            gen_norms = gen_embeds / (np.linalg.norm(gen_embeds, axis=1, keepdims=True) + 1e-8)
            all_answer_rel.append(float(np.mean(np.dot(gen_norms, q_norm))))
        else:
            all_answer_rel.append(0.0)

        # Context Relevance
        relevant_sents = await _extract_relevant(llm, question, context[:8000])
        total_sents = len([s for s in re.split(r'[。！？\n]+', context) if s.strip()])
        if total_sents > 0 and relevant_sents:
            all_context_rel.append(min(len(relevant_sents) / total_sents, 1.0))
        else:
            all_context_rel.append(0.0)

    return {
        "avg_faithfulness": float(np.mean(all_faith)) if all_faith else 0.0,
        "avg_answer_relevance": float(np.mean(all_answer_rel)) if all_answer_rel else 0.0,
        "avg_context_relevance": float(np.mean(all_context_rel)) if all_context_rel else 0.0,
    }


def _parse_json(text: str) -> dict | list:
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _parse_json_array(text: str) -> list:
    result = _parse_json(text)
    return result if isinstance(result, list) else []


async def _extract_statements(llm, answer: str) -> list[str]:
    try:
        msg = await llm.ainvoke([HumanMessage(content=FAITHFULNESS_STATEMENTS_PROMPT.format(answer=answer))])
        return _parse_json_array(msg.content)
    except Exception:
        return []


async def _verify_statement(llm, statement: str, context: str) -> str:
    try:
        msg = await llm.ainvoke([HumanMessage(content=FAITHFULNESS_VERIFY_PROMPT.format(statement=statement, context=context))])
        result = _parse_json(msg.content)
        return result.get("verdict", "NO") if isinstance(result, dict) else "NO"
    except Exception:
        return "NO"


async def _reverse_questions(llm, answer: str) -> list[str]:
    try:
        msg = await llm.ainvoke([HumanMessage(content=ANSWER_RELEVANCE_REVERSE_PROMPT.format(answer=answer))])
        return _parse_json_array(msg.content)
    except Exception:
        return []


async def _extract_relevant(llm, question: str, context: str) -> list[str]:
    try:
        msg = await llm.ainvoke([HumanMessage(content=CONTEXT_RELEVANCE_EXTRACT_PROMPT.format(question=question, context=context))])
        return _parse_json_array(msg.content)
    except Exception:
        return []


def _build_report_dict(report: BenchmarkReport, vr: VersionResult) -> dict:
    """构建单个版本的报告字典（用于 JSON 快照）。"""
    r = report.retrieval
    return {
        "run_id": report.run_id,
        "agent_version": vr.agent_version,
        "dataset_name": report.dataset_name,
        "question_count": report.question_count,
        "git_commit": report.git_commit,
        "timing": {
            "avg_total_ms": vr.timing.avg_total_ms,
            "avg_ttft_ms": vr.timing.avg_ttft_ms,
            "avg_retrieval_ms": vr.timing.avg_retrieval_ms,
            "avg_generation_ms": vr.timing.avg_generation_ms,
        },
        "retrieval": {
            "recall_at_5": r.avg_recall.get(5, 0) if r else 0,
            "precision_at_5": r.avg_precision.get(5, 0) if r else 0,
            "mrr": r.avg_mrr if r else 0,
            "ndcg_at_5": r.avg_ndcg.get(5, 0) if r else 0,
            "hit_at_5": r.avg_hit.get(5, 0) if r else 0,
        },
        "ragas": {
            "faithfulness": vr.ragas_faithfulness,
            "answer_relevance": vr.ragas_answer_relevance,
            "context_relevance": vr.ragas_context_relevance,
        },
        "passed": report.passed_map.get(vr.agent_version, False),
    }
