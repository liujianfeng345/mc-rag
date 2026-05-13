"""
评测数据集构建辅助工具。

两种模式：

1. 基础模式（默认）：从知识库采样文档片段，用 LLM 生成候选问题
   - 输出仅包含 question 字段
   - 需要人工补充 relevant_sources 和 golden_answer
   - 适用于 RAGAS 评测（不需要标注）和自动检索评测（--auto）
   - uv run python -m src.eval.dataset_builder -c 10 -o eval_data/generated.json

2. 完整模式（--full）：从知识库采样文档片段，用 LLM 生成完整标注
   - 输出包含 question + relevant_sources + golden_answer 三个字段
   - 直接可用于精确检索评测
   - uv run python -m src.eval.dataset_builder -c 10 --full -o eval_data/full_dataset.json
"""

import argparse
import asyncio
import json
import random
import re
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

文档片段（来源: {source}）：
{chunk}

请用 JSON 数组格式输出，每个元素是一个字符串（问题）。只输出 JSON。

示例输出格式：
["问题1", "问题2", "问题3"]"""


GOLDEN_ANSWER_PROMPT = """你是一个 Minecraft 开发文档助手。请根据给定的文档片段，回答用户问题。

要求：
1. 答案只依据给定的文档片段，不要编造
2. 如果文档片段信息不完整，只回答能回答的部分
3. 用中文简洁回答，200 字以内

文档片段：
{chunk}

用户问题：{question}

请直接输出答案，不要加前缀或后缀标记。"""


def _parse_json_array(text: str) -> list:
    """从 LLM 输出中解析 JSON 数组。"""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


async def _gen_questions_for_chunks(
    chunks: list[dict],
    llm,
) -> list[dict]:
    """
    用 LLM 为每个文档片段生成候选问题。

    返回：[{"question": "...", "source": "path/to/file.md"}, ...]
    """
    results = []
    for entry in chunks:
        try:
            msg = await llm.ainvoke([
                HumanMessage(content=QUESTION_GEN_PROMPT.format(
                    chunk=entry["content"][:3000],
                    source=entry["source"],
                ))
            ])
            questions = _parse_json_array(msg.content)
            for q in questions:
                results.append({
                    "question": q,
                    "source": entry["source"],
                })
        except Exception as e:
            print(f"  生成问题失败 [{entry['source']}]: {e}")
            continue
    return results


async def _gen_golden_answer(question: str, chunk: str, llm) -> str:
    """用 LLM 根据文档片段生成参考答案。"""
    try:
        msg = await llm.ainvoke([
            HumanMessage(content=GOLDEN_ANSWER_PROMPT.format(
                chunk=chunk[:3000],
                question=question,
            ))
        ])
        return msg.content.strip()
    except Exception:
        return ""


async def _sample_chunks(count: int) -> list[dict]:
    """从 ChromaDB 随机采样文档片段。"""
    store = VectorStore()
    stats = await store.stats()
    print(f"  知识库总块数: {stats['文档块数量']}")

    collection_count = store.collection.count()
    if collection_count == 0:
        print("错误: 知识库为空，请先执行 build 命令")
        return []

    result = store.collection.get(
        limit=min(count, collection_count),
        include=["documents", "metadatas"],
        offset=random.randint(0, max(collection_count - count, 0)),
    )

    chunks = []
    for i in range(len(result["ids"])):
        meta = result["metadatas"][i] or {}
        source = meta.get("source", "未知")
        content = result["documents"][i] or ""
        if content.strip():
            chunks.append({"content": content, "source": source})

    return chunks


async def build_dataset(
    count: int = 10,
    output_path: str = "eval_data/generated_questions.json",
    full_mode: bool = False,
) -> None:
    """
    构建评测数据集。

    参数：
        count: 采样文档片段数量
        output_path: 输出 JSON 文件路径
        full_mode: 是否生成完整标注（question + relevant_sources + golden_answer）
    """
    mode_label = "完整模式（含标注）" if full_mode else "基础模式（仅问题）"
    print(f"评测数据集构建 - {mode_label}")
    print(f"从知识库采样 {count} 个文档片段...")

    chunks = await _sample_chunks(count)
    if not chunks:
        return

    print(f"  实际采样 {len(chunks)} 个片段，开始生成问题...")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=2048,
    )

    # 生成问题
    qa_pairs = await _gen_questions_for_chunks(chunks, llm)

    # 按来源去重：同一来源最多保留 3 个问题
    seen_sources: dict[str, int] = {}
    deduped = []
    for pair in qa_pairs:
        src = pair["source"]
        if seen_sources.get(src, 0) < 3:
            deduped.append(pair)
            seen_sources[src] = seen_sources.get(src, 0) + 1

    # 问题级别去重
    seen_questions = set()
    unique_pairs = []
    for pair in deduped:
        if pair["question"] not in seen_questions:
            unique_pairs.append(pair)
            seen_questions.add(pair["question"])

    print(f"  生成 {len(qa_pairs)} 个候选问题，去重后保留 {len(unique_pairs)} 个")

    # 构建数据集
    dataset = []

    if full_mode:
        print("  生成参考答案...")
        # 先建立 source → chunk 的映射
        source_chunk_map = {c["source"]: c["content"] for c in chunks}

        for idx, pair in enumerate(unique_pairs):
            chunk = source_chunk_map.get(pair["source"], "")
            golden = await _gen_golden_answer(pair["question"], chunk, llm)
            dataset.append({
                "question": pair["question"],
                "relevant_sources": [pair["source"]],
                "golden_answer": golden,
            })
            if (idx + 1) % 5 == 0:
                print(f"    进度: {idx + 1}/{len(unique_pairs)}")
    else:
        for pair in unique_pairs:
            dataset.append({
                "question": pair["question"],
                "relevant_sources": [],
                "golden_answer": "",
            })

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n生成完成: {len(dataset)} 个问题")
    print(f"输出文件: {output_path}")

    if full_mode:
        print(f"\n下一步（完整模式）：")
        print(f"  1. 抽查数据集，确认问题质量和答案准确性")
        print(f"  2. 运行精确检索评测:")
        print(f"     uv run python -m src.main eval -d {output_path} --retrieval-only")
    else:
        print(f"\n下一步（基础模式）：")
        print(f"  1. 直接用 RAGAS 评测（无需标注）:")
        print(f"     uv run python -m src.main eval -d {output_path} --ragas-only")
        print(f"  2. 或用自动检索评测（LLM 判定相关性）:")
        print(f"     uv run python -m src.main eval -d {output_path} --retrieval-only --auto")
        print(f"  3. 如需精确检索评测，请手动补充 relevant_sources 字段")


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
    parser.add_argument(
        "--full",
        action="store_true",
        help="生成完整标注数据集（question + relevant_sources + golden_answer）",
    )
    args = parser.parse_args()

    asyncio.run(build_dataset(
        count=args.count,
        output_path=args.output,
        full_mode=args.full,
    ))
