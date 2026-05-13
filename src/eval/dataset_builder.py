"""
评测数据集构建辅助工具。

功能：
1. 从知识库随机采样文档片段，用 LLM 生成候选问题
2. 用户手动筛选和修正后保存为正式评测集

使用方式：
    uv run python -m src.eval.dataset_builder --count 10 --output eval_data/generated.json
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
from ..vector.vector_store import VectorStore


QUESTION_GEN_PROMPT = """你是一个测试问题生成器。给定一段 Minecraft 开发文档，请生成 3 个用户可能向 RAG 系统提问的问题。

要求：
1. 问题应该自然、像真实用户会问的
2. 问题应该能通过给定的文档片段来回答
3. 包含不同类型的问题：简单的定义类、操作步骤类、配置参数类

文档片段：
{chunk}

请用 JSON 数组格式输出，每个元素是一个字符串（问题）。只输出 JSON。

示例输出格式：
["问题1", "问题2", "问题3"]"""


async def generate_questions(doc_chunks: list[str], llm) -> list[str]:
    """用 LLM 从文档片段生成候选问题。"""
    all_questions = []
    for chunk in doc_chunks:
        try:
            msg = await llm.ainvoke([
                HumanMessage(content=QUESTION_GEN_PROMPT.format(chunk=chunk[:3000]))
            ])
            text = msg.content.strip()
            # 去除可能的 markdown 代码块包装
            import re
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
            questions = json.loads(text)
            if isinstance(questions, list):
                all_questions.extend(questions)
        except Exception as e:
            print(f"  生成问题失败: {e}")
            continue
    return all_questions


async def build_dataset(
    count: int = 10,
    output_path: str = "eval_data/generated_questions.json",
) -> None:
    """
    从知识库采样文档片段，用 LLM 生成候选评测问题。

    参数：
        count: 采样文档片段数量
        output_path: 输出 JSON 文件路径
    """
    print(f"从知识库采样 {count} 个文档片段...")

    store = VectorStore()
    stats = await store.stats()
    print(f"  知识库总块数: {stats['文档块数量']}")

    # 从 ChromaDB 随机采样
    collection_count = store.collection.count()
    if collection_count == 0:
        print("错误: 知识库为空，请先执行 build 命令")
        return

    # 随机选取 count 个 ID
    result = store.collection.get(
        limit=min(count, collection_count),
        include=["documents", "metadatas"],
        offset=random.randint(0, max(collection_count - count, 0)),
    )

    chunks = result["documents"] or []

    print(f"  实际采样 {len(chunks)} 个片段，开始生成问题...")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=2048,
    )

    questions = await generate_questions(chunks, llm)

    # 去重并整理
    unique_questions = list(dict.fromkeys(questions))  # 保序去重

    # 构建评测数据集格式
    dataset = []
    for q in unique_questions:
        dataset.append({
            "question": q,
            "relevant_sources": [],  # 需要手动补充
            "golden_answer": "",     # 需要手动补充
        })

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n生成完成: {len(dataset)} 个候选问题")
    print(f"输出文件: {output_path}")
    print(f"\n下一步:")
    print(f"  1. 手动审查并删除不合适的题目")
    print(f"  2. 为每个问题填充 relevant_sources（标注相关文档路径）")
    print(f"  3. 运行评测: uv run python -m src.main eval -d {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测数据集构建辅助工具")
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=10,
        help="采样文档片段数量（默认 10）",
    )
    parser.add_argument(
        "--output", "-o",
        default="eval_data/generated_questions.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    asyncio.run(build_dataset(count=args.count, output_path=args.output))
