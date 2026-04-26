"""节点函数"""

from ..utils.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    RETRIEVAL_TOP_K,
)
from .state import RAGState, DocumentRelevance
from .prompt import GRADE_PROMPT, SYSTEM_PROMPT, REWRITE_PROMPT
from ..vector.vector_store import VectorStore

from langchain_deepseek import ChatDeepSeek
from langchain.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)


def create_llm(
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> ChatDeepSeek:
    """创建大模型"""
    llm = ChatDeepSeek(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    return llm


async def retrieve_node(state: RAGState, vector_store: VectorStore) -> dict:
    """
    检索节点：根据用户问题从向量数据库检索相关文档。

    首次检索使用原始问题；重写后使用重写的问题。
    """
    question = state['current_query'] if state.get('current_query') else state["question"]

    documents = await vector_store.search(question, top_k=RETRIEVAL_TOP_K)

    return {
        "documents": documents,
    }


async def grade_node(state: RAGState) -> dict:
    """
    评分节点：评估检索到的文档是否与问题相关。

    使用 LLM 对每个文档片段逐一评分，只有"相关"的文档才会进入生成阶段。
    """
    question = state['current_query'] if state.get('current_query') else state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {"documents": []}

    llm = create_llm(temperature=0)
    llm_structured = llm.with_structured_output(DocumentRelevance)
    relevant_docs = []

    for doc in documents:
        prompt = GRADE_PROMPT.format(
            question=question,
            document=doc.page_content[:2000],
        )
        response: DocumentRelevance = await llm_structured.ainvoke(
            [HumanMessage(content=prompt)]
        )
        try:
            grade = response.relevant
        except:
            grade = "相关"
        if grade == "相关":
            relevant_docs.append(doc)

    return {
        "documents": relevant_docs,
    }


async def generate_node(state: RAGState) -> dict:
    """
    生成节点：基于相关文档生成最终答案。

    输出包含引用来源，便于用户追溯。
    """
    question = state["question"]
    documents = state.get("documents", [])

    # 构建上下文（带来源标注）
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "未知")
        folder = doc.metadata.get("folder", "")
        context_parts.append(
            f"[文档{i}] 来源: {source}\n分类: {folder}\n内容:\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = SYSTEM_PROMPT.format(context=context)
    llm = create_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    # 流式调用 LLM，收集完整回复（token 级别事件由上层 astream_events 捕获）
    full_content = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_content += chunk.content

    return {
        "generation": full_content,
        "messages": [AIMessage(content=full_content)]
    }


async def rewrite_node(state: RAGState) -> dict:
    """
    重写节点：当检索到的文档不相关时，优化查询表达。

    使用 LLM 将模糊问题改写为更具体、更适合检索的形式。
    限制最多重写 2 次，防止无限循环。
    """
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0) + 1

    llm = create_llm(temperature=0.3)
    prompt = REWRITE_PROMPT.format(question=question)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    rewritten = response.content.strip()

    return {
        "messages": [HumanMessage(content=rewritten)],
        "rewrite_count": rewrite_count,
        "current_query": rewritten,
    }
