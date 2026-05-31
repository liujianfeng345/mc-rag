"""Benchmark CLI 入口，供 main.py 调用。"""

import asyncio
import sys

from rich.console import Console

# 确保 Windows 控制台能正确输出 Unicode 字符（如 ✓）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .benchmark_config import DEFAULT_DATASET, DEFAULT_VERSIONS
from .database import BenchmarkDB
from .reporter import print_report, save_json_report, print_history
from .runner import BenchmarkRunner


async def run_benchmark(
    dataset: str = "",
    versions: str = "",
    profile_only: bool = False,
    eval_only: bool = False,
    set_baseline: bool = False,
    history: bool = False,
) -> None:
    """benchmark 命令分发入口。"""
    console = Console(force_terminal=True)

    if set_baseline:
        db = BenchmarkDB()
        db.init_default_baselines()
        console.print("[green]✓ 基线阈值已写入/更新[/green]")
        baselines = db.get_baselines()
        for name, cfg in baselines.items():
            console.print(f"  {name}: min={cfg['min_threshold']}, max_degrade={cfg['max_degradation_pct']}%")
        return

    if history:
        db = BenchmarkDB()
        version = versions.split(",")[0].strip() if versions else ""
        print_history(console, db, version=version)
        return

    dataset_path = dataset or DEFAULT_DATASET
    version_list = [v.strip() for v in versions.split(",") if v.strip()] if versions else DEFAULT_VERSIONS

    console.print(f"[bold cyan]=== Benchmark 开始 ===[/bold cyan]")
    console.print(f"数据集: {dataset_path}")
    console.print(f"版本: {', '.join(version_list)}")
    console.print(f"模式: {'仅性能' if profile_only else '仅评测' if eval_only else '完整'}\n")

    runner = BenchmarkRunner(
        dataset_path=dataset_path,
        versions=version_list,
        profile_only=profile_only,
        eval_only=eval_only,
    )
    report = await runner.run()

    print_report(report, console)
    json_path = save_json_report(report)
    console.print(f"[green]JSON 报告已保存: {json_path}[/green]")
