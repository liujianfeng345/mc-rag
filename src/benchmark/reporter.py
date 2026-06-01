# src/benchmark/reporter.py
"""报告输出 — Rich 终端表格 + JSON 导出 + 历史趋势。"""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .benchmark_config import REPORT_DIR
from .database import BenchmarkDB
from .runner import BenchmarkReport, VersionResult


def print_report(report: BenchmarkReport, console: Console | None = None) -> None:
    """打印终端 benchmark 报告（版本对比表 + 趋势）。"""
    console = console or Console()

    console.print(Panel(
        f"[bold]Benchmark Report {report.run_id}[/bold]\n"
        f"Dataset: {report.dataset_name} | {report.question_count} questions\n"
        f"Commit: {report.git_commit}",
        border_style="cyan",
    ))

    # ---- 版本对比表 ----
    table = Table(title="版本对比")
    table.add_column("Metric", style="cyan")
    for vr in report.versions:
        table.add_column(vr.agent_version.upper(), justify="right")

    # 检索指标行
    r = report.retrieval
    if r:
        _add_metric_row(table, "Recall@5", [r.avg_recall.get(5, 0)] * len(report.versions), fmt=".1%")
        _add_metric_row(table, "MRR", [r.avg_mrr] * len(report.versions), fmt=".3f")
        _add_metric_row(table, "Hit@5", [r.avg_hit.get(5, 0)] * len(report.versions), fmt=".1%")
        _add_metric_row(table, "NDCG@5", [r.avg_ndcg.get(5, 0)] * len(report.versions), fmt=".3f")

    # 分隔线
    table.add_row("─" * 16, *(["─" * 10] * len(report.versions)))

    # RAGAS 指标行
    _add_metric_row(table, "Faithfulness", [vr.ragas_faithfulness for vr in report.versions], fmt=".1%")
    _add_metric_row(table, "Answer Relevance", [vr.ragas_answer_relevance for vr in report.versions], fmt=".1%")
    _add_metric_row(table, "Context Relevance", [vr.ragas_context_relevance for vr in report.versions], fmt=".1%")

    # 分隔线
    table.add_row("─" * 16, *(["─" * 10] * len(report.versions)))

    # 性能指标行
    _add_metric_row(table, "Avg TTFT (ms)", [vr.timing.avg_ttft_ms for vr in report.versions], fmt=".0f")
    _add_metric_row(table, "Avg Total (ms)", [vr.timing.avg_total_ms for vr in report.versions], fmt=".0f")

    # 分隔线
    table.add_row("─" * 16, *(["─" * 10] * len(report.versions)))

    # 通过状态行
    passed_vals = ["✓" if report.passed_map.get(vr.agent_version, False) else "✗ FAIL" for vr in report.versions]
    _add_metric_row(table, "Passed", passed_vals, fmt="")

    console.print(table)

    # ---- 难度分层对比表 ----
    diffs = ["简单", "中等", "复杂"]
    for diff in diffs:
        has_data = any(
            vr.ragas_by_difficulty.get(diff, {}).get("faithfulness", 0) > 0
            for vr in report.versions
        )
        if not has_data:
            continue
        dt = Table(title=f"难度: {diff}")
        dt.add_column("指标", style="cyan")
        for vr in report.versions:
            dt.add_column(vr.agent_version.upper(), justify="right")

        _add_metric_row(dt, "Faithfulness",
            [vr.ragas_by_difficulty.get(diff, {}).get("faithfulness", 0) for vr in report.versions],
            fmt=".1%")
        _add_metric_row(dt, "Answer Relevance",
            [vr.ragas_by_difficulty.get(diff, {}).get("answer_relevance", 0) for vr in report.versions],
            fmt=".1%")
        _add_metric_row(dt, "Context Relevance",
            [vr.ragas_by_difficulty.get(diff, {}).get("context_relevance", 0) for vr in report.versions],
            fmt=".1%")

        console.print()
        console.print(dt)

    # ---- 趋势摘要 ----
    console.print("\n[bold]趋势: 本次 vs 上次[/bold]")
    db = BenchmarkDB()
    for vr in report.versions:
        last = db.get_last_run(vr.agent_version, report.dataset_name)
        trend = _build_trend_lines(vr, last, report.retrieval)
        console.print(f"  [cyan]{vr.agent_version}[/cyan]: {trend or '（首次运行，无历史数据）'}")

    console.print()


def _add_metric_row(table: Table, name: str, values: list, fmt: str = ".3f") -> None:
    """格式化一行指标到表格。"""
    formatted = []
    for v in values:
        if isinstance(v, str):
            formatted.append(v)
        elif fmt:
            formatted.append(f"{v:{fmt}}")
        else:
            formatted.append(str(v))
    table.add_row(name, *formatted)


def _build_trend_lines(vr: VersionResult, last_run: dict | None, retrieval) -> str:
    """构建趋势描述文字。"""
    if last_run is None:
        return ""
    lines = []
    # Faithfulness
    prev_f = last_run.get("faithfulness", 0)
    if prev_f:
        delta = vr.ragas_faithfulness - prev_f
        arrow = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "→"
        lines.append(f"Faithfulness: {vr.ragas_faithfulness:.1%} ({arrow}{abs(delta):.2f})")
    return ", ".join(lines)


def save_json_report(report: BenchmarkReport) -> str:
    """导出完整 JSON 报告到文件。"""
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{report.run_id}.json"

    data = {
        "run_id": report.run_id,
        "dataset_name": report.dataset_name,
        "question_count": report.question_count,
        "git_commit": report.git_commit,
        "total_elapsed_ms": report.total_elapsed_ms,
        "versions": {},
    }
    for vr in report.versions:
        data["versions"][vr.agent_version] = {
            "timing": {
                "avg_total_ms": vr.timing.avg_total_ms,
                "avg_ttft_ms": vr.timing.avg_ttft_ms,
                "avg_generation_ms": vr.timing.avg_generation_ms,
            },
            "ragas": {
                "faithfulness": vr.ragas_faithfulness,
                "answer_relevance": vr.ragas_answer_relevance,
                "context_relevance": vr.ragas_context_relevance,
                "by_difficulty": vr.ragas_by_difficulty,
            },
            "passed": report.passed_map.get(vr.agent_version, False),
        }

    if report.retrieval:
        data["retrieval"] = {
            "recall_at_5": report.retrieval.avg_recall.get(5, 0),
            "precision_at_5": report.retrieval.avg_precision.get(5, 0),
            "mrr": report.retrieval.avg_mrr,
            "ndcg_at_5": report.retrieval.avg_ndcg.get(5, 0),
            "hit_at_5": report.retrieval.avg_hit.get(5, 0),
        }

    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filepath)


def print_history(console: Console, db: BenchmarkDB, version: str = "", limit: int = 10) -> None:
    """打印历史趋势表格。"""
    rows = db.get_history(agent_version=version, limit=limit)
    if not rows:
        console.print("[dim]暂无历史记录[/dim]")
        return

    title = f"历史趋势 ({version or '全部版本'})"
    table = Table(title=title)
    table.add_column("Run ID", style="dim")
    table.add_column("Version", style="cyan")
    table.add_column("Recall@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("TTFT(ms)", justify="right")
    table.add_column("Pass", justify="center")

    for row in rows:
        passed = "✓" if row["passed"] else "✗"
        table.add_row(
            str(row["id"]),
            str(row["agent_version"]),
            f"{row['recall_at_5']:.1%}",
            f"{row['mrr']:.3f}",
            f"{row['faithfulness']:.1%}",
            f"{row['avg_ttft_ms']:.0f}",
            passed,
        )
    console.print(table)
