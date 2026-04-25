"""
文档加载与分块模块

负责：
1. 从 data/ 目录递归加载所有 Markdown 文件
2. 解析 YAML Front Matter（难度、时间等元数据）
3. 使用 RecursiveCharacterTextSplitter 按 Markdown 结构分块
4. 为每个块生成带有文件路径、标题层级的元数据
"""

import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


def parse_front_matter(content: str) -> tuple[dict, str]:
    """
    解析 Markdown 文件的 YAML Front Matter。

    Front Matter 格式：
    ---
    key: value
    ---
    正文内容...

    返回 (元数据字典, 正文内容)
    """
    fm_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = fm_pattern.match(content)
    if not match:
        return {}, content

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}

    body = content[match.end() :]
    return metadata, body


def extract_heading_hierarchy(content: str) -> list[str]:
    """提取文档的标题层级，用于在元数据中保留结构上下文。"""
    headings = []
    for line in content.split("\n"):
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            headings.append(f"{'#' * level} {heading_text}")
    return headings


def load_markdown_files(base_dir: str = None) -> list[dict]:
    """
    递归加载所有 Markdown 文件。

    每个文件返回一个字典：
    {
        "content": str,           # 去除了 front matter 的正文
        "metadata": {             # 文件级元数据
            "source": str,        # 相对于 data/ 的路径
            "file_name": str,     # 文件名
            "headings": list,     # 标题层级
            "front_matter": dict, # front matter 原始数据
        }
    }
    """
    base = Path(base_dir or DOCS_DIR)
    documents = []

    for md_file in sorted(base.rglob("*.md")):
        # 跳过图片目录中的 md 文件（如果有的话）
        if "images" in md_file.parts or "picture" in md_file.parts:
            continue

        try:
            raw = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        fm_meta, body = parse_front_matter(raw)
        headings = extract_heading_hierarchy(body)

        rel_path = md_file.relative_to(base)

        documents.append(
            {
                "content": body.strip(),
                "metadata": {
                    "source": str(rel_path),
                    "file_name": md_file.name,
                    "headings": "; ".join(headings[:5]),  # 只保留前5级标题
                    "front_matter": fm_meta,
                    "folder": str(rel_path.parent),
                },
            }
        )

    return documents


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    创建针对 Markdown 文档优化的文本分割器。

    使用自定义分隔符列表，优先在章节边界分割，保持文档结构的完整性。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        keep_separator=True,          # 保留分隔符（标题）以便上下文理解
        strip_whitespace=True,
        length_function=len,
        is_separator_regex=False,
    )


def load_and_split_documents(base_dir: str = None) -> list[Document]:
    """
    加载所有 Markdown 文件并分割为 LangChain Document 对象。

    每个 Document 的元数据包含：
    - source: 源文件路径
    - file_name: 文件名
    - headings: 标题层级上下文
    - folder: 所属文件夹（分类信息）
    - chunk_index: 块索引（用于引用定位）

    Returns:
        list[Document]: 分割后的文档块列表
    """
    raw_docs = load_markdown_files(base_dir)
    splitter = create_text_splitter()

    all_documents = []
    for raw in raw_docs:
        file_docs = splitter.create_documents(
            texts=[raw["content"]],
            metadatas=[raw["metadata"]],
        )
        # 为每个块添加索引
        for i, doc in enumerate(file_docs):
            doc.metadata["chunk_index"] = i
        all_documents.extend(file_docs)

    return all_documents


# =============================================================================
# 便捷函数：直接从目录加载文档（用于 CLI）
# =============================================================================
def get_document_stats(base_dir: str = None) -> dict:
    """获取文档统计信息。"""
    raw = load_markdown_files(base_dir)
    docs = load_and_split_documents(base_dir)
    folders = set(d["metadata"]["folder"] for d in raw)
    return {
        "文件数": len(raw),
        "文档块数": len(docs),
        "分类目录数": len(folders),
        "总字符数": sum(len(d["content"]) for d in raw),
        "平均块大小": sum(len(d.page_content) for d in docs) // max(len(docs), 1),
    }
