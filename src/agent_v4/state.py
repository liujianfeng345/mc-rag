"""状态定义"""

from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class AgentState(TypedDict):
    """深度 RAG 代理的状态

    messages:       对话历史（自动合并同角色消息）
    question:       用户原始问题
    research_plan:  分解后的子问题列表
    findings:       各子问题的研究发现（含主题、内容、来源）
    final_report:   最终生成的回答
    documents:      引用的文档列表（用于来源展示）
    """

    messages: Annotated[list, add_messages]
    question: str
    research_plan: list[str]
    findings: list[dict]
    final_report: str
    documents: list[Document]
    reflection: dict                         # 反思结果（is_sufficient, gaps, queries）
    refine_count: int                        # 已精炼次数（初始 0，上限 2）
    supplemental_findings: list[dict]        # 补充检索的研究发现


# =============================================================================
# 结构化输出模型
# =============================================================================
class ResearchPlan(BaseModel):
    """研究计划"""
    sub_questions: list[str] = Field(
        description="分解后的子问题列表，每个聚焦一个方面，共2-4个"
    )


class ReflectionResult(BaseModel):
    """反思评估结果"""
    is_sufficient: bool = Field(
        description="答案是否充分满足用户需求。全部事实有支撑、所有方面已覆盖则为 True"
    )
    factual_issues: list[str] = Field(
        description="事实性问题列表。每个条目描述一个无法在检索文档中找到支撑的断言"
    )
    coverage_gaps: list[str] = Field(
        description="覆盖缺口列表。用户问题中被遗漏或未充分回答的方面"
    )
    follow_up_queries: list[str] = Field(
        description="针对缺口的补充检索查询。每个查询应具体、可直接用于知识库检索。"
                    "若答案已充分则为空列表。上限 3 条。"
    )
