"""
RAGAS 风格生成质量评测 —— 用 LLM 做裁判评估答案质量。

评估三个维度：

1. Faithfulness（忠实度）：答案中的每句话是否能从检索到的文档中得到支撑
   - 先让 LLM 从答案中提取所有"独立陈述"
   - 再让 LLM 逐条判断每个陈述能否在上下文中找到依据
   - 得分 = 有依据的陈述数 / 总陈述数，取值 0~1

2. Answer Relevance（答案相关性）：答案是否紧扣问题
   - 让 LLM 根据答案反向生成"可能的用户问题"
   - 计算生成的问题与原始问题的语义相似度（用嵌入向量）
   - 得分 = 余弦相似度均值，取值 0~1

3. Context Relevance（上下文相关性）：检索到的文档是否与问题相关
   - 让 LLM 从上下文中提取"与问题相关的句子"
   - 得分 = 相关句子数 / 上下文总句子数，取值 0~1

使用方式：
    from src.eval.ragas_eval import RAGASEvaluator
    from src.vector.vector_store import VectorStore

    store = VectorStore()
    evaluator = RAGASEvaluator(store)
    result = await evaluator.evaluate(dataset)
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from .dataset import EvalDataset
from ..vector.vector_store import VectorStore
from ..utils.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    RETRIEVAL_TOP_K,
)


# =============================================================================
# 用于评测的 Prompt（中文，适配中文知识库）
# =============================================================================

FAITHFULNESS_STATEMENTS_PROMPT = """你的任务是从给定的答案中提取所有"独立的事实陈述"。

独立事实陈述的判断标准：
- 每一句话应该是一个可以独立验证真伪的主张
- 把复合句拆成简单句
- 忽略"你好"、"希望对你有帮助"这类礼貌用语

答案：
{answer}

请用 JSON 数组格式输出，每个元素是一个字符串（陈述句）。只输出 JSON，不要加其他内容。

示例输出格式：
["陈述句1", "陈述句2", "陈述句3"]"""


FAITHFULNESS_VERIFY_PROMPT = """你的任务是判断一个"陈述"是否能在给定的"参考文档"中找到依据。

判断标准：
- 如果陈述的信息在参考文档中有明确提及或可以合理推断 → 标记为 "YES"
- 如果陈述的信息在参考文档中找不到任何支撑 → 标记为 "NO"
- 如果陈述与参考文档中的信息矛盾 → 标记为 "CONTRADICTION"

参考文档：
{context}

陈述：{statement}

请用 JSON 格式回答：{{"verdict": "YES/NO/CONTRADICTION", "reason": "简短说明"}}"""


ANSWER_RELEVANCE_REVERSE_PROMPT = """给定一个回答，请你反向生成这个回答可能在回答的"用户问题"。

回答：
{answer}

请生成 3 个不同措辞的可能问题。用 JSON 数组格式输出，只输出 JSON。

示例输出格式：
["问题1", "问题2", "问题3"]"""


CONTEXT_RELEVANCE_EXTRACT_PROMPT = """你的任务是从给定的文档上下文中，提取所有"与问题直接相关的句子"。

问题：{question}

文档上下文：
{context}

请用 JSON 数组格式输出相关的句子（只提取确实与问题相关的句子，不要提取无关内容）。

示例输出格式：
["相关句子1", "相关句子2"]"""


@dataclass
class RAGASMetrics:
    """单条问答的生成质量指标。"""
    question: str
    difficulty: str = ""  # 来自 EvalItem.difficulty
    faithfulness: float = 0.0          # 忠实度 0~1
    answer_relevance: float = 0.0     # 答案相关性 0~1
    context_relevance: float = 0.0    # 上下文相关性 0~1
    answer: str = ""
    error: str = ""


@dataclass
class RAGASResult:
    """完整 RAGAS 评测结果。"""
    dataset_name: str
    metrics: list[RAGASMetrics]

    @property
    def avg_faithfulness(self) -> float:
        vals = [m.faithfulness for m in self.metrics if m.error == ""]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def avg_answer_relevance(self) -> float:
        vals = [m.answer_relevance for m in self.metrics if m.error == ""]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def avg_context_relevance(self) -> float:
        vals = [m.context_relevance for m in self.metrics if m.error == ""]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def by_difficulty(self) -> dict[str, "RAGASResult"]:
        """按难度分组返回各自的 RAGASResult。"""
        groups: dict[str, list[RAGASMetrics]] = {"简单": [], "中等": [], "复杂": [], "未分类": []}
        for m in self.metrics:
            key = m.difficulty or "未分类"
            groups.setdefault(key, []).append(m)
        result: dict[str, "RAGASResult"] = {}
        for diff, metrics_list in groups.items():
            if not metrics_list:
                continue
            result[diff] = RAGASResult(
                dataset_name=f"{self.dataset_name} ({diff})",
                metrics=metrics_list,
            )
        return result

    def rich_table(self) -> Table:
        table = Table(title=f"生成质量评测: {self.dataset_name}")
        table.add_column("问题", style="cyan", max_width=40)
        table.add_column("Faithfulness", justify="right")
        table.add_column("Answer Rel.", justify="right")
        table.add_column("Context Rel.", justify="right")

        for m in self.metrics:
            if m.error:
                table.add_row(
                    m.question[:40],
                    f"[red]错误[/red]",
                    f"[red]错误[/red]",
                    f"[red]错误[/red]",
                )
            else:
                color_f = "green" if m.faithfulness >= 0.7 else "yellow" if m.faithfulness >= 0.4 else "red"
                color_a = "green" if m.answer_relevance >= 0.7 else "yellow" if m.answer_relevance >= 0.4 else "red"
                color_c = "green" if m.context_relevance >= 0.7 else "yellow" if m.context_relevance >= 0.4 else "red"
                table.add_row(
                    m.question[:40],
                    f"[{color_f}]{m.faithfulness:.1%}[/{color_f}]",
                    f"[{color_a}]{m.answer_relevance:.1%}[/{color_a}]",
                    f"[{color_c}]{m.context_relevance:.1%}[/{color_c}]",
                )

        table.add_row("─" * 40, "─" * 8, "─" * 8, "─" * 8)
        table.add_row(
            "[bold]平均值[/bold]",
            f"[bold]{self.avg_faithfulness:.1%}[/bold]",
            f"[bold]{self.avg_answer_relevance:.1%}[/bold]",
            f"[bold]{self.avg_context_relevance:.1%}[/bold]",
        )
        return table


class RAGASEvaluator:
    """RAGAS 风格评测器。

    对每条评测数据：
    1. 执行 hybrid_search 检索相关文档
    2. 调用 RAG 的 synthesize 生成答案
    3. 用 LLM 裁判评估三个维度的质量

    使用方式：
        store = VectorStore()
        evaluator = RAGASEvaluator(store)
        result = await evaluator.evaluate(dataset)
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

        # 评测专用 LLM（temperature=0 保证一致性）
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
            max_tokens=2048,
        )

        # 嵌入模型（用于 answer relevance 的语义相似度计算）
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 用于生成答案的 LLM（可以稍有温度）
        self.answer_llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=4096,
        )

    async def evaluate(
        self,
        dataset: EvalDataset,
        verbose: bool = True,
    ) -> RAGASResult:
        """对数据集执行 RAGAS 风格评测。"""
        console = Console() if verbose else None
        all_metrics: list[RAGASMetrics] = []

        for idx, item in enumerate(dataset.items):
            if console:
                console.print(f"[dim][{idx + 1}/{len(dataset)}] 评测: {item.question[:80]}...[/dim]")

            try:
                metrics = await self._evaluate_single(item)
                metrics.difficulty = item.difficulty  # 补写难度
                all_metrics.append(metrics)
            except Exception as e:
                all_metrics.append(RAGASMetrics(
                    question=item.question,
                    difficulty=item.difficulty,
                    error=str(e),
                ))
                if console:
                    console.print(f"  [red]错误: {e}[/red]")

        result = RAGASResult(
            dataset_name=dataset.name,
            metrics=all_metrics,
        )

        if console:
            console.print(result.rich_table())

        return result

    async def _evaluate_single(self, item) -> RAGASMetrics:
        """评测单条数据。"""
        from .dataset import EvalItem

        # 1. 检索相关文档
        docs = await self.store.hybrid_search(item.question, top_k=RETRIEVAL_TOP_K)
        context = "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )

        # 2. 生成答案
        answer_msg = await self.answer_llm.ainvoke([
            SystemMessage(content=(
                "你是一个 Minecraft 开发文档助手。请根据给定的参考文档回答用户问题。"
                "如果文档中没有相关信息，请如实说明。不要编造内容。"
            )),
            HumanMessage(content=f"参考文档:\n{context}\n\n问题: {item.question}\n\n请用中文回答："),
        ])
        answer = answer_msg.content

        # 3. 计算三个指标（并行以提高速度）
        faithfulness, answer_relevance, context_relevance = await asyncio.gather(
            self._compute_faithfulness(answer, context),
            self._compute_answer_relevance(item.question, answer),
            self._compute_context_relevance(item.question, context),
        )

        return RAGASMetrics(
            question=item.question,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_relevance=context_relevance,
            answer=answer,
        )

    async def _compute_faithfulness(self, answer: str, context: str) -> float:
        """计算忠实度分数。

        1. 从答案中提取独立陈述
        2. 对每条陈述，判断是否能从上下文中找到依据
        3. 得分 = 有依据的陈述 / 总陈述数
        """
        # Step 1: 提取陈述
        statements = await self._llm_extract_statements(answer)
        if not statements:
            return 1.0  # 没有可提取的陈述，默认忠实

        # Step 2: 逐条验证
        verdicts = await asyncio.gather(*[
            self._llm_verify_statement(stmt, context) for stmt in statements
        ])

        # Step 3: 计算得分
        yes_count = sum(1 for v in verdicts if v == "YES")
        return yes_count / len(statements)

    async def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """计算答案相关性分数。

        1. 从答案反向生成可能的用户问题
        2. 计算生成的问题与原始问题的语义余弦相似度
        3. 得分 = 相似度均值
        """
        # 反向生成问题
        generated = await self._llm_reverse_questions(answer)
        if not generated:
            return 0.0

        # 计算嵌入向量
        all_texts = [question] + generated
        all_embeddings = await asyncio.to_thread(
            self.embeddings.embed_documents, all_texts
        )

        q_embed = np.array(all_embeddings[0])
        gen_embeds = np.array(all_embeddings[1:])

        # 余弦相似度
        q_norm = q_embed / (np.linalg.norm(q_embed) + 1e-8)
        gen_norms = gen_embeds / (np.linalg.norm(gen_embeds, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(gen_norms, q_norm)

        return float(np.mean(similarities))

    async def _compute_context_relevance(self, question: str, context: str) -> float:
        """计算上下文相关性分数。

        1. 让 LLM 从上下文中提取与问题相关的句子
        2. 得分 = 相关句子数 / 上下文总句子数
        """
        relevant = await self._llm_extract_relevant_sentences(question, context)
        if not relevant:
            return 0.0

        # 估算上下文的总句子数
        total_sentences = len([s for s in re.split(r'[。！？\n]+', context) if s.strip()])
        if total_sentences == 0:
            return 0.0

        return min(len(relevant) / total_sentences, 1.0)

    # ----- LLM 辅助方法 -----

    async def _llm_extract_statements(self, answer: str) -> list[str]:
        """从答案中提取独立陈述。"""
        try:
            msg = await self.llm.ainvoke([
                HumanMessage(content=FAITHFULNESS_STATEMENTS_PROMPT.format(answer=answer))
            ])
            return self._parse_json_array(msg.content)
        except Exception:
            return []

    async def _llm_verify_statement(self, statement: str, context: str) -> str:
        """验证一条陈述是否能从上下文中找到依据。"""
        try:
            msg = await self.llm.ainvoke([
                HumanMessage(content=FAITHFULNESS_VERIFY_PROMPT.format(
                    statement=statement, context=context
                ))
            ])
            result = self._parse_json(msg.content)
            return result.get("verdict", "NO") if isinstance(result, dict) else "NO"
        except Exception:
            return "NO"

    async def _llm_reverse_questions(self, answer: str) -> list[str]:
        """反向生成可能的用户问题。"""
        try:
            msg = await self.llm.ainvoke([
                HumanMessage(content=ANSWER_RELEVANCE_REVERSE_PROMPT.format(answer=answer))
            ])
            return self._parse_json_array(msg.content)
        except Exception:
            return []

    async def _llm_extract_relevant_sentences(self, question: str, context: str) -> list[str]:
        """从上下文中提取与问题相关的句子。"""
        try:
            # 限制上下文长度避免 token 超限
            truncated = context[:8000]
            msg = await self.llm.ainvoke([
                HumanMessage(content=CONTEXT_RELEVANCE_EXTRACT_PROMPT.format(
                    question=question, context=truncated
                ))
            ])
            return self._parse_json_array(msg.content)
        except Exception:
            return []

    @staticmethod
    def _parse_json_array(text: str) -> list:
        """从 LLM 输出中解析 JSON 数组。"""
        if not text:
            return []
        # 去除可能的 markdown 代码块包装
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        try:
            result = json.loads(text)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        """从 LLM 输出中解析 JSON 对象。"""
        if not text:
            return {}
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
