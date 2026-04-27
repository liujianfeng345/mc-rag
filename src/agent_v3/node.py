"""节点函数

核心流程：decompose（分解问题）→ research_batch（并行研究）→ synthesize（综合合成）
每个研究员在 ReAct 循环中自主决定检索时机和检索角度。
"""

import asyncio

from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..utils.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    RETRIEVAL_TOP_K,
)
from .state import AgentState
from .prompt import DECOMPOSE_PROMPT, RESEARCHER_SYSTEM_PROMPT, SYNTHESIZE_PROMPT
from ..vector.vector_store import VectorStore


# =============================================================================
# LLM 工厂
# =============================================================================
def create_llm(
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> ChatDeepSeek:
    """创建 DeepSeek 大模型实例"""
    return ChatDeepSeek(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


# =============================================================================
# 结构化输出模型
# =============================================================================
class ResearchPlan(BaseModel):
    """研究计划"""
    sub_questions: list[str] = Field(
        description="分解后的子问题列表，每个聚焦一个方面，共2-4个"
    )


# =============================================================================
# 工具定义
# =============================================================================
@tool
async def think_tool(thought: str) -> str:
    """记录思考过程或规划下一步检索策略。在检索前后使用此工具来梳理研究思路。"""
    return f"思考已记录：{thought}"


def _make_retrieve_tool(vs: VectorStore) -> StructuredTool:
    """创建知识库检索工具（工厂函数，注入向量存储依赖）"""

    async def _retrieve(query: str) -> str:
        """在 Minecraft 开发知识库中检索与查询相关的文档。返回格式化的文档内容和来源信息。"""
        docs = await vs.hybrid_search(query, top_k=RETRIEVAL_TOP_K)
        return _format_retrieved_docs(docs)

    return StructuredTool.from_function(
        name="retrieve_from_kb",
        description="在 Minecraft 开发知识库中搜索相关文档。输入具体的搜索查询语句，返回匹配的文档内容以及来源信息。",
        coroutine=_retrieve,
    )


# =============================================================================
# 工具函数
# =============================================================================
def _format_retrieved_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为 LLM 可读的文本"""
    if not docs:
        return "未找到相关文档，请尝试使用不同的关键词或换一种表述重新检索。"
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "未知")
        folder = doc.metadata.get("folder", "")
        parts.append(
            f"[文档{i}] 来源：{src}\n分类：{folder}\n内容：\n{doc.page_content[:1500]}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# 研究员 ReAct 循环（内部函数，非图节点）
# =============================================================================
async def _run_researcher(
    topic: str,
    vector_store: VectorStore,
    max_iterations: int = 5,
) -> dict:
    """
    为单个研究主题运行 ReAct 循环。

    研究员拥有 retrieve_from_kb 和 think_tool 两种工具，
    可以自主决定何时检索、检索什么内容，直到收集到足够信息。

    返回: {"topic": str, "findings": str, "sources": [str], "documents": [Document]}
    """
    retrieve_tool = _make_retrieve_tool(vector_store)
    tools = [retrieve_tool, think_tool]

    llm = create_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"请研究以下主题并输出结构化的技术发现：\n\n{topic}"),
    ]

    collected_docs: list[Document] = []

    for _ in range(max_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # 无工具调用 → 研究员认为信息已足够，输出即为研究发现
        if not response.tool_calls:
            return {
                "topic": topic,
                "findings": response.content,
                "sources": _extract_sources(messages),
                "documents": collected_docs,
            }

        # 执行工具调用
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            if tool_name == "retrieve_from_kb":
                # 仅检索一次，同时用于工具结果和文档收集
                query = tool_args.get("query", "")
                docs = await vector_store.hybrid_search(
                    query, top_k=RETRIEVAL_TOP_K
                )
                collected_docs.extend(docs)
                result = _format_retrieved_docs(docs)
            elif tool_name == "think_tool":
                result = await think_tool.ainvoke(tool_args)
            else:
                result = f"未知工具：{tool_name}"

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
            ))

    # 达到最大迭代次数，强制要求总结
    messages.append(HumanMessage(content="请基于以上所有检索结果，总结你的研究发现。"))
    final_response = await llm.ainvoke(messages)

    return {
        "topic": topic,
        "findings": final_response.content,
        "sources": _extract_sources(messages),
        "documents": collected_docs,
    }


def _extract_sources(messages: list) -> list[str]:
    """从工具消息中提取文档来源名称（去重排序）"""
    sources = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = str(msg.content)
        for line in content.split("\n"):
            if "来源：" in line:
                source = line.split("来源：")[-1].strip()
                if source and source != "未知":
                    sources.add(source)
    return sorted(sources)


# =============================================================================
# 图节点
# =============================================================================
async def decompose_node(state: AgentState) -> dict:
    """
    分解节点：将用户问题分解为多个子问题。

    对于简单问题可能只产出 1 个子问题，复杂问题产出 2-4 个。
    """
    question = _get_question(state)
    if not question:
        return {"research_plan": []}

    llm = create_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(ResearchPlan)

    prompt = DECOMPOSE_PROMPT.format(question=question)
    try:
        response: ResearchPlan = await structured_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )
        plan = response.sub_questions
    except Exception:
        # 结构化输出失败时回退：直接用原问题
        plan = [question]

    return {
        "question": question,
        "research_plan": plan,
    }


async def research_batch_node(
    state: AgentState, vector_store: VectorStore
) -> dict:
    """
    并行研究节点：为每个子问题启动独立研究员，并行执行。

    每个研究员在 ReAct 循环中自主进行多轮、多角度知识库检索。
    """
    plan = state.get("research_plan", [])
    question = state.get("question", "")

    # 如果分解失败，直接用原问题研究
    topics = plan if plan else [question]

    # 并行启动所有研究员
    tasks = [_run_researcher(topic, vector_store) for topic in topics]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    findings = []
    all_docs: list[Document] = []

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            findings.append({
                "topic": topics[i] if i < len(topics) else "未知",
                "findings": f"研究过程出错：{r}",
                "sources": [],
            })
        else:
            findings.append({
                "topic": r["topic"],
                "findings": r["findings"],
                "sources": r.get("sources", []),
            })
            all_docs.extend(r.get("documents", []))

    # 文档去重
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc.metadata.get("source", "") + str(
            doc.metadata.get("chunk_index", "")
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return {
        "findings": findings,
        "documents": unique_docs,
    }


async def synthesize_node(state: AgentState) -> dict:
    """
    综合节点：汇总所有研究发现，生成结构化最终回答。

    使用流式输出以改善用户体验，token 级别事件由上层
    astream_events 捕获。
    """
    question = _get_question(state)
    findings = state.get("findings", [])

    # 构建研究发现文本
    parts = []
    for i, f in enumerate(findings, 1):
        topic = f.get("topic", f"研究方向 {i}")
        content = f.get("findings", "无结果")
        sources = f.get("sources", [])
        parts.append(f"### 研究方向 {i}：{topic}\n{content}")
        if sources:
            parts.append(f"参考来源：{', '.join(sources)}")
    findings_text = "\n\n".join(parts) if parts else "未找到相关研究结果。"

    prompt = SYNTHESIZE_PROMPT.format(question=question, findings=findings_text)

    llm = create_llm()

    full_content = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            full_content += chunk.content

    return {
        "final_report": full_content,
        "messages": [AIMessage(content=full_content)],
    }


def _get_question(state: AgentState) -> str:
    """获取用户问题（兼容 LangSmith 回放时 state 中无 question 字段的情况）"""
    question = state.get("question", "")
    if question:
        return question
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict):
                return first.get("text", "")
            return str(first)
    return ""
