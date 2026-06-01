"""
评测数据集定义与加载。

数据集格式（JSON）：
[
  {
    "question": "如何自定义武器属性？",
    "relevant_sources": ["data/20-玩法开发/15-自定义武器/README.md"],
    "golden_answer": "通过修改武器配置文件中的 attributes 字段..."
  }
]

字段说明：
- question（必填）：测试问题
- relevant_sources（可选）：标注的相关文档路径列表，用于检索评测
- golden_answer（可选）：参考答案，用于展示但非必需
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalItem:
    """单条评测数据。"""
    question: str
    relevant_sources: list[str] = field(default_factory=list)
    golden_answer: str = ""
    difficulty: str = ""  # "简单" | "中等" | "复杂" | ""

    @classmethod
    def from_dict(cls, data: dict) -> "EvalItem":
        return cls(
            question=data["question"],
            relevant_sources=data.get("relevant_sources", []),
            golden_answer=data.get("golden_answer", ""),
            difficulty=data.get("difficulty", ""),
        )

    def to_dict(self) -> dict:
        """序列化为字典，用于回写 JSON。"""
        d: dict = {"question": self.question}
        if self.relevant_sources:
            d["relevant_sources"] = self.relevant_sources
        if self.golden_answer:
            d["golden_answer"] = self.golden_answer
        if self.difficulty:
            d["difficulty"] = self.difficulty
        return d


class EvalDataset:
    """评测数据集，从 JSON 文件加载。"""

    def __init__(self, items: list[EvalItem], name: str = "", source_path: str = ""):
        self.items = items
        self.name = name
        self.source_path = source_path

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        """从 JSON 文件加载评测数据集。"""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"评测数据集不存在: {path}")

        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("评测数据集必须是 JSON 数组格式")

        items = [EvalItem.from_dict(item) for item in raw]
        return cls(items, name=file_path.stem, source_path=str(file_path.resolve()))

    def save_to_json(self, path: str = "") -> None:
        """将数据集（含难度标签）写回 JSON 文件。不传 path 则覆盖源文件。"""
        target = Path(path) if path else Path(self.source_path)
        data = [item.to_dict() for item in self.items]
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def has_relevance_labels(self) -> bool:
        """是否包含相关文档标注（有标注才能做检索评测）。"""
        return any(item.relevant_sources for item in self.items)

    @property
    def has_golden_answers(self) -> bool:
        """是否包含参考答案。"""
        return any(item.golden_answer for item in self.items)

    async def classify_difficulty(self) -> None:
        """对数据集中未分类的题目进行 LLM 难度分类。

        已有 difficulty 标签的题目会被跳过，避免重复调用 LLM。
        分类结果写入 EvalItem.difficulty，可随后调用 save_to_json() 持久化。
        """
        unclassified = [item for item in self.items if not item.difficulty]
        if not unclassified:
            return
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from ..utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

        PROMPT = """你的任务是判断一个"RAG 系统的用户问题"的难度等级。

判断标准（只看问题本身的推理深度和涉及的知识点数量）：
- 简单：单一知识点，直接查询文档即可回答，不需要推理
- 中等：涉及 2-3 个知识点，需要一定推理或跨段落信息整合
- 复杂：涉及多个知识点、需要多步推理、涉及计算或需要跨文档整合信息

问题：{question}

请只输出一个词：简单、中等 或 复杂。"""

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
            max_tokens=32,
        )

        async def _classify_one(item: EvalItem) -> None:
            try:
                msg = await llm.ainvoke([
                    HumanMessage(content=PROMPT.format(question=item.question))
                ])
                text = msg.content.strip()
                if "复杂" in text:
                    item.difficulty = "复杂"
                elif "中等" in text:
                    item.difficulty = "中等"
                elif "简单" in text:
                    item.difficulty = "简单"
            except Exception:
                pass  # 分类失败保持 ""

        await asyncio.gather(*[_classify_one(item) for item in unclassified])

    @property
    def difficulty_distribution(self) -> dict[str, int]:
        """返回各难度题目数量分布。"""
        dist: dict[str, int] = {}
        for item in self.items:
            key = item.difficulty or "未分类"
            dist[key] = dist.get(key, 0) + 1
        return dist

    def stats(self) -> dict:
        """数据集的统计信息。"""
        return {
            "数据集名称": self.name,
            "问题数量": len(self.items),
            "有相关文档标注": sum(1 for i in self.items if i.relevant_sources),
            "有参考答案": sum(1 for i in self.items if i.golden_answer),
        }
