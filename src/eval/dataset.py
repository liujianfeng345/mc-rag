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

    @classmethod
    def from_dict(cls, data: dict) -> "EvalItem":
        return cls(
            question=data["question"],
            relevant_sources=data.get("relevant_sources", []),
            golden_answer=data.get("golden_answer", ""),
        )


class EvalDataset:
    """评测数据集，从 JSON 文件加载。"""

    def __init__(self, items: list[EvalItem], name: str = ""):
        self.items = items
        self.name = name

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
        return cls(items, name=file_path.stem)

    @property
    def has_relevance_labels(self) -> bool:
        """是否包含相关文档标注（有标注才能做检索评测）。"""
        return any(item.relevant_sources for item in self.items)

    @property
    def has_golden_answers(self) -> bool:
        """是否包含参考答案。"""
        return any(item.golden_answer for item in self.items)

    def stats(self) -> dict:
        """数据集的统计信息。"""
        return {
            "数据集名称": self.name,
            "问题数量": len(self.items),
            "有相关文档标注": sum(1 for i in self.items if i.relevant_sources),
            "有参考答案": sum(1 for i in self.items if i.golden_answer),
        }
