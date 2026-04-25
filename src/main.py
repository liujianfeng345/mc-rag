"""
RAG 系统主入口

支持两种模式：
1. 构建索引：uv run mc-rag build
2. 交互问答：uv run mc-rag ask "问题"
3. Web 演示：uv run mc-rag demo
"""

import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from .utils.config import DOCS_DIR
from .vector.document_loader import load_and_split_documents, get_document_stats
from .vector.vector_store import VectorStore
from .agent.graph import build_rag_graph

console = Console()


# =============================================================================
# 子命令
# =============================================================================
def cmd_build():
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
    with console.status("[bold yellow]加载并分割文档...[/bold yellow]"):
        documents = load_and_split_documents(DOCS_DIR)
    console.print(f"[green]✓ 文档已分割为 {len(documents)} 个块[/green]")

    # 3. 构建向量索引
    store = VectorStore()
    store.reset()  # 重建索引

    with console.status("[bold yellow]生成嵌入向量并建立索引...[/bold yellow]"):
        added = store.add_documents(documents)
    console.print(f"[green]✓ 已索引 {added} 个文档块[/green]")

    # 4. 显示存储统计
    s = store.stats()
    console.print(f"\n[bold green]✅ 索引构建完成！[/bold green]")
    console.print(f"   集合: {s['集合名称']}")
    console.print(f"   块数: {s['文档块数量']}")
    console.print(f"   路径: {s['存储路径']}")


def cmd_ask(question: str):
    """单次问答"""
    store = VectorStore()
    graph = build_rag_graph(store)

    console.print(f"\n[bold cyan]🔍 问题: {question}[/bold cyan]\n")

    with console.status("[bold yellow]处理中...[/bold yellow]"):
        result = graph.invoke({"question": question, "rewrite_count": 0})

    # 显示答案
    answer = result.get("generation", "无法生成答案")
    console.print(Panel(Markdown(answer), title="回答", border_style="green"))

    # 显示引用
    if result.get("documents"):
        console.print("\n[bold]📄 参考文档:[/bold]")
        for doc in result["documents"]:
            source = doc.metadata.get("source", "未知")
            score = doc.metadata.get("_score", 0)
            console.print(f"  • {source} (相似度: {score:.3f})")


def cmd_demo():
    """交互式问答循环"""
    store = VectorStore()
    graph = build_rag_graph(store)

    console.print(Panel(
        "[bold]Minecraft 开发文档 RAG 助手[/bold]\n\n"
        "输入问题即可获取答案，输入 [cyan]/quit[/cyan] 退出\n"
        f"知识库: {DOCS_DIR}\n"
        f"文档块数: {store.stats()['文档块数量']}",
        border_style="cyan",
    ))

    session_messages = []

    while True:
        try:
            question = console.input("\n[bold cyan]你: [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "退出"):
            console.print("[yellow]再见！[/yellow]")
            break

        with console.status("[dim]思考中...[/dim]"):
            result = graph.invoke({
                "question": question,
                "messages": session_messages,
                "rewrite_count": 0,
            })

        answer = result.get("generation", "无法生成答案")
        console.print(Panel(Markdown(answer), title="助手", border_style="green"))

        # 显示引用
        if result.get("documents"):
            sources = set()
            for doc in result["documents"]:
                sources.add(doc.metadata.get("source", "未知"))
            if sources:
                console.print("[dim]📄 参考: " + ", ".join(list(sources)[:3]) + "[/dim]")


# =============================================================================
# CLI
# =============================================================================
def main():
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
        cmd_build()
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "demo":
        cmd_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
