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
from .prompt import GRADE_PROMPT

from langchain_openai import ChatOpenAI
from langchain.messages import (
    HumanMessage,
)

def create_llm(
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> ChatOpenAI:
    """创建大模型"""
    llm = ChatOpenAI(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    return llm


def retrieve_node(state: RAGState, vector_store) -> dict:
    """
    检索节点：根据用户问题从向量数据库检索相关文档。

    首次检索使用原始问题；重写后使用重写的问题。
    """
    question = state["question"]

    # 如果有历史重写，使用最后一条 HumanMessage 作为查询
    if state.get("messages"):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

    documents = vector_store.search(question, top_k=RETRIEVAL_TOP_K)

    return {
        "documents": documents,
    }

def grade_node(state: RAGState) -> dict:
    """
    评分节点：评估检索到的文档是否与问题相关。

    使用 LLM 对每个文档片段逐一评分，只有"相关"的文档才会进入生成阶段。
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {"documents": []}
    
    llm = create_llm(temperature=0)
    llm_structured = llm.with_structured_output(DocumentRelevance)
    relevant_docs = []

    for doc in documents:
        prompt = GRADE_PROMPT.format(
            question=question,
            document=doc.page_content[:2000], # 控制token消耗
        )
        response: DocumentRelevance = llm_structured.invoke([HumanMessage(content=prompt)])
        grade = response.relevant
        if grade == "相关":
            relevant_docs.append(doc)

    if not relevant_docs:
        relevant_docs = documents

    return {
        documents: relevant_docs,
    }
