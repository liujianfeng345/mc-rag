"""节点函数

核心流程：decompose（分解问题）→ research_batch（并行研究）→ synthesize（综合合成）
每个研究员在 ReAct 循环中自主决定检索时机和检索角度。
"""

import asyncio

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
    REFLECT_TEMPERATURE,
    MAX_REFINE_ITERATIONS,
)
from .state import AgentState, ResearchPlan, ReflectionResult
from .prompt import (
    DECOMPOSE_PROMPT, RESEARCHER_SYSTEM_PROMPT,
    SYNTHESIZE_PROMPT, REFLECT_PROMPT,
    REFINE_SYNTHESIZE_PROMPT, SUPPLEMENT_PROMPT,
)
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


def _extract_sources_from_docs(docs: list[Document]) -> list[str]:
    """从文档列表中提取来源文件名（去重排序）"""
    sources = set()
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src and src != "未知":
            sources.add(src)
    return sorted(sources)


def _build_doc_summaries(documents: list[Document]) -> str:
    """构建可供 LLM 检查的文档来源与内容摘要"""
    if not documents:
        return "（无检索文档）"
    parts = []
    seen = set()
    for doc in documents:
        src = doc.metadata.get("source", "未知")
        if src not in seen:
            seen.add(src)
            folder = doc.metadata.get("folder", "")
            snippet = doc.page_content[:300].replace("\n", " ")
            parts.append(f"- {src}（{folder}）: {snippet}...")
    return "\n".join(parts)


def _format_findings(findings: list[dict]) -> str:
    """将研究发现列表格式化为反思节点可读的文本"""
    if not findings:
        return "（无研究发现）"
    parts = []
    for i, f in enumerate(findings, 1):
        topic = f.get("topic", f"研究方向 {i}")
        content = f.get("findings", "无结果")
        sources = f.get("sources", [])
        part = f"### {topic}\n{content}"
        if sources:
            part += f"\n来源：{', '.join(sources)}"
        parts.append(part)
    return "\n\n".join(parts)


def _merge_deduplicate(
    existing_docs: list[Document], new_docs: list[Document]
) -> list[Document]:
    """合并新旧文档列表并去重（按 source + chunk_index 判重）"""
    merged = list(existing_docs)
    seen = set()
    for doc in merged:
        key = doc.metadata.get("source", "") + str(
            doc.metadata.get("chunk_index", "")
        )
        seen.add(key)
    for doc in new_docs:
        key = doc.metadata.get("source", "") + str(
            doc.metadata.get("chunk_index", "")
        )
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged


async def _generate_supplement(
    docs: list[Document], query: str, gaps: list[str]
) -> str:
    """用 LLM 从补充检索文档中提取关键信息"""
    if not docs:
        return "补充检索无结果，文档中未找到相关内容。"

    docs_text = _format_retrieved_docs(docs)
    gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "（未指定）"

    prompt = SUPPLEMENT_PROMPT.format(
        query=query,
        gaps=gaps_text,
        docs_text=docs_text,
    )

    llm = create_llm(temperature=0.0)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


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

    首次合成（refine_count == 0）：使用 SYNTHESIZE_PROMPT 从零生成
    精炼后合成（refine_count > 0）：使用 REFINE_SYNTHESIZE_PROMPT 修订已有答案
    """
    question = _get_question(state)
    primary_findings = state.get("findings", [])
    supplemental = state.get("supplemental_findings", [])
    refine_count = state.get("refine_count", 0)

    if refine_count == 0:
        # 首次合成：从零生成
        parts = []
        for i, f in enumerate(primary_findings, 1):
            topic = f.get("topic", f"研究方向 {i}")
            content = f.get("findings", "无结果")
            sources = f.get("sources", [])
            parts.append(f"### 研究方向 {i}：{topic}\n{content}")
            if sources:
                parts.append(f"参考来源：{', '.join(sources)}")
        findings_text = "\n\n".join(parts) if parts else "未找到相关研究结果。"

        prompt = SYNTHESIZE_PROMPT.format(question=question, findings=findings_text)
    else:
        # 精炼后合成：修订已有答案
        previous_answer = state.get("final_report", "")
        gaps = state.get("reflection", {}).get("coverage_gaps", [])

        primary_text = _format_findings(primary_findings)
        supplemental_text = _format_findings(supplemental)
        gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "（未指定）"

        prompt = REFINE_SYNTHESIZE_PROMPT.format(
            question=question,
            previous_answer=previous_answer,
            identified_gaps=gaps_text,
            primary_findings=primary_text,
            supplemental_findings=supplemental_text,
        )

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


async def reflect_node(state: AgentState) -> dict:
    """
    反思节点：系统性评估已生成答案的质量。

    返回:
        reflection: {is_sufficient, factual_issues, coverage_gaps, follow_up_queries}
        refine_count: 不变（由 refine 节点递增）
    """
    question = state["question"]
    final_report = state["final_report"]
    findings = state.get("findings", [])
    documents = state.get("documents", [])

    # 构建可供 LLM 检查的文档摘要
    doc_summaries = _build_doc_summaries(documents)
    findings_text = _format_findings(findings)

    llm = create_llm(temperature=REFLECT_TEMPERATURE)  # 反思用低温度，减少随机性
    structured_llm = llm.with_structured_output(ReflectionResult)

    prompt = REFLECT_PROMPT.format(
        question=question,
        final_report=final_report,
        findings_summary=findings_text,
        doc_sources=doc_summaries,
    )

    try:
        response: ReflectionResult = await structured_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )
    except Exception:
        # 结构化输出失败，降级为直接通过
        return {
            "reflection": {
                "is_sufficient": True,
                "factual_issues": [],
                "coverage_gaps": [],
                "follow_up_queries": [],
            }
        }

    return {
        "reflection": {
            "is_sufficient": response.is_sufficient,
            "factual_issues": response.factual_issues,
            "coverage_gaps": response.coverage_gaps,
            "follow_up_queries": response.follow_up_queries,
        }
    }


async def refine_node(state: AgentState, vector_store: VectorStore) -> dict:
    """
    精炼节点：针对反思发现的缺口进行定向补充检索。

    流程：
        1. 对每个 follow_up_query 执行 hybrid_search
        2. 合并新文档（与已有文档去重）
        3. 用 LLM 为每组新文档撰写补充研究发现
        4. 追加到 supplemental_findings

    返回:
        supplemental_findings: 追加的补充发现
        documents: 扩充后的文档列表（去重合并）
        refine_count: +1
    """
    reflection = state.get("reflection", {})
    follow_up_queries = reflection.get("follow_up_queries", [])
    existing_docs = state.get("documents", [])
    supplementary_findings = state.get("supplemental_findings", [])

    # 1. 并行执行所有补充检索
    tasks = [
        vector_store.hybrid_search(q, top_k=RETRIEVAL_TOP_K)
        for q in follow_up_queries
    ]
    results = await asyncio.gather(*tasks)

    # 2. 为每个查询生成补充发现
    new_findings = []
    all_new_docs = []

    for query, docs in zip(follow_up_queries, results):
        if docs:
            all_new_docs.extend(docs)
            # 用 LLM 从新文档中提取关键信息
            summary = await _generate_supplement(docs, query, reflection.get("coverage_gaps", []))
            new_findings.append({
                "topic": f"[补充] {query}",
                "findings": summary,
                "sources": _extract_sources_from_docs(docs),
            })

    # 3. 文档去重合并
    merged_docs = _merge_deduplicate(existing_docs, all_new_docs)

    return {
        "supplemental_findings": supplementary_findings + new_findings,
        "documents": merged_docs,
        "refine_count": state.get("refine_count", 0) + 1,
    }
