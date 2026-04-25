"""
RAG 系统主入口

支持两种模式：
1. 构建索引：uv run mc-rag build
2. 交互问答：uv run mc-rag ask "问题"
3. Web 演示：uv run mc-rag demo
"""

import argparse
import asyncio

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .utils.config import DOCS_DIR
from .vector.document_loader import load_and_split_documents, get_document_stats
from .vector.vector_store import VectorStore
from .agent.graph import build_rag_graph

console = Console()


# =============================================================================
# 子命令
# =============================================================================
async def cmd_build():
    """构建/重建向量索引"""
    console.print("[bold cyan]📚 开始构建文档索引...[/bold cyan]\n")

    # 1. 统计文档
    stats = get_document_stats(DOCS_DIR)
    table = Table(title="文档统计")
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)

    # 2. 加载并分割
    console.print("[bold yellow]正在加载并分割文档...[/bold yellow]")
    documents = await asyncio.to_thread(load_and_split_documents, DOCS_DIR)
    console.print(f"[green]✓ 文档已分割为 {len(documents)} 个块[/green]")

    store = VectorStore()
    await store.reset()  # 重建索引

    console.print("[bold yellow]正在生成嵌入向量并建立索引...[/bold yellow]")
    added = await store.add_documents(documents)
    console.print(f"[green]✓ 已索引 {added} 个文档块[/green]")

    # 4. 显示存储统计
    s = await store.stats()
    console.print(f"\n[bold green]✅ 索引构建完成！[/bold green]")
    console.print(f"   集合: {s['集合名称']}")
    console.print(f"   块数: {s['文档块数量']}")
    console.print(f"   路径: {s['存储路径']}")


async def cmd_ask(question: str):
    """单次问答（流式输出）"""
    store = VectorStore()
    graph = build_rag_graph(store)

    console.print(f"\n[bold cyan]🔍 问题: {question}[/bold cyan]\n")

    console.print("[bold green]回答:[/bold green]")
    console.print("─" * 60)

    # 收集最终状态用于显示引用
    final_state: dict = {}

    # 使用 astream_events 实现流式 token 输出
    async for event in graph.astream_events(
        {"question": question, "rewrite_count": 0},
        version="v2",
    ):
        kind = event["event"]

        # 流式打印 LLM 生成的每个 token
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                console.print(chunk.content, end="")

        # 捕获节点输出，取最后一次作为最终状态
        if kind == "on_chain_end" and isinstance(event.get("data", {}).get("output"), dict):
            final_state.update(event["data"]["output"])

    console.print("\n" + "─" * 60)

    # 显示引用
    documents = final_state.get("documents", [])
    if documents:
        console.print("\n[bold]📄 参考文档:[/bold]")
        for doc in documents:
            source = doc.metadata.get("source", "未知")
            score = doc.metadata.get("_score", 0)
            console.print(f"  • {source} (相似度: {score:.3f})")


async def cmd_demo():
    """交互式问答循环"""
    store = VectorStore()
    graph = build_rag_graph(store)

    console.print(
        Panel(
            "[bold]Minecraft 开发文档 RAG 助手[/bold]\n\n"
            "输入问题即可获取答案，输入 [cyan]/quit[/cyan] 退出\n"
            f"知识库: {DOCS_DIR}\n"
            f"文档块数: {(await store.stats())['文档块数量']}",
            border_style="cyan",
        )
    )

    session_messages = []

    while True:
        try:
            question = await asyncio.to_thread(
                lambda: console.input("\n[bold cyan]你: [/bold cyan]").strip()
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "退出"):
            console.print("[yellow]再见！[/yellow]")
            break

        console.print("[dim]思考中...[/dim]")

        # 流式输出
        console.print("[bold green]助手:[/bold green]")
        console.print("─" * 60)
        final_state: dict = {}

        async for event in graph.astream_events(
            {
                "question": question,
                "messages": session_messages,
                "rewrite_count": 0,
            },
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    console.print(chunk.content, end="")

            if kind == "on_chain_end" and isinstance(event.get("data", {}).get("output"), dict):
                final_state.update(event["data"]["output"])

        console.print("\n" + "─" * 60)

        # 显示引用
        documents = final_state.get("documents", [])
        if documents:
            sources = set()
            for doc in documents:
                sources.add(doc.metadata.get("source", "未知"))
            if sources:
                console.print(
                    "[dim]📄 参考: " + ", ".join(list(sources)[:3]) + "[/dim]"
                )


# =============================================================================
# CLI
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(
        prog="mc-rag",
        description="Minecraft 开发文档 RAG 知识库系统",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # build
    subparsers.add_parser("build", help="构建/重建文档向量索引")

    # ask
    ask_parser = subparsers.add_parser("ask", help="单次问答")
    ask_parser.add_argument("question", help="要查询的问题")

    # demo
    subparsers.add_parser("demo", help="交互式问答")

    args = parser.parse_args()

    if args.command == "build":
        await cmd_build()
    elif args.command == "ask":
        await cmd_ask(args.question)
    elif args.command == "demo":
        await cmd_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
