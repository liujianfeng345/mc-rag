"""Web API 层 — 封装 RAG 图为可流式调用的生成器

支持 v1-v4 所有版本，通过 agent_version 参数动态切换。
供 Streamlit app.py 调用，也可被其他 Web 框架复用。
"""

from typing import AsyncIterator, Optional

from dotenv import load_dotenv

from ..vector.vector_store import VectorStore

load_dotenv()

# 最大输入长度（安全限制）
MAX_INPUT_LENGTH = 500


def validate_input(question: str) -> Optional[str]:
    """校验用户输入，返回错误信息或 None"""
    if not question or not question.strip():
        return "请输入问题"
    if len(question) > MAX_INPUT_LENGTH:
        return f"问题过长（最多 {MAX_INPUT_LENGTH} 字符）"
    return None


def _build_graph(version: str, vector_store: VectorStore):
    """按版本构建 RAG 图（动态导入，避免循环引用）"""
    if version == "v4":
        from ..agent_v4.graph import build_rag_graph
    elif version == "v3":
        from ..agent_v3.graph import build_rag_graph
    elif version == "v2":
        from ..agent_v2.graph import build_rag_graph
    else:
        from ..agent.graph import build_rag_graph
    return build_rag_graph(vector_store)


async def stream_rag_response(
    question: str,
    vector_store: VectorStore,
    history: Optional[list] = None,
    agent_version: str = "v4",
) -> AsyncIterator[dict]:
    """流式 RAG 生成器，逐 token yield 并在结束时回传元数据。

    每步 yield:
        {"type": "token", "content": "文"}      流式 token
        {"type": "done", "answer": "...", ...}  完成信号（含来源文档列表）
    """
    graph = _build_graph(agent_version, vector_store)

    input_state: dict = {"question": question}
    if agent_version in ("v1", "v2"):
        input_state["rewrite_count"] = 0
    if history:
        input_state["messages"] = history

    final_state: dict = {}

    async for event in graph.astream_events(input_state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {"type": "token", "content": chunk.content}

        if kind == "on_chain_end" and isinstance(
            event.get("data", {}).get("output"), dict
        ):
            final_state.update(event["data"]["output"])

    # 从最终状态提取答案和来源
    answer = final_state.get("final_report") or final_state.get("generation", "")
    documents = final_state.get("documents", [])

    sources: list[dict] = []
    seen = set()
    for doc in documents:
        src = doc.metadata.get("source", "未知")
        if src not in seen:
            seen.add(src)
            score = doc.metadata.get("_rrf_score") or doc.metadata.get("_score", 0)
            sources.append({"source": src, "score": round(float(score), 3)})

    # 记录最终答案到消息历史（不带工具调用的 AI 消息）
    from langchain_core.messages import AIMessage

    history_msg = AIMessage(content=answer)

    yield {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "history_msg": history_msg,
    }
