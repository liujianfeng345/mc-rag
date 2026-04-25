"""
向量存储模块

基于 ChromaDB 实现持久化向量存储，支持：
1. 文档嵌入与索引
2. 语义检索
3. 集合管理（重置、统计）
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..utils.config import (
    VECTOR_DB_DIR,
    VECTOR_COLLECTION,
    RETRIEVAL_TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
)


def _sanitize_metadata(metadata: dict) -> dict:
    """将 metadata 中不可序列化的值（dict/list）转为 JSON 字符串，以符合 ChromaDB 约束。"""
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = value
    return sanitized


def _restore_metadata(metadata: dict) -> dict:
    """将 JSON 字符串值尝试还原为原始类型（dict/list），反向操作 _sanitize_metadata。"""
    restored = {}
    for key, value in metadata.items():
        if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
            try:
                restored[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                restored[key] = value
        else:
            restored[key] = value
    return restored


class VectorStore:
    """
    ChromaDB 向量存储封装。

    使用方式：
        store = VectorStore()
        store.add_documents(docs)       # 索引文档
        results = store.search("问题")   # 检索
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
    ):
        persist_dir = persist_dir or VECTOR_DB_DIR
        self.collection_name = collection_name or VECTOR_COLLECTION
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化嵌入模型（多语言模型，中文友好）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 初始化 ChromaDB 持久化客户端
        self._client = PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        # 过滤已存在的文档（幂等性）
        existing_ids = set(self._collection.get()["ids"])
        new_docs = []
        new_ids = []

        for doc in documents:
            doc_id = f"{doc.metadata['source']}:{doc.metadata['chunk_index']}"
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)

        if not new_docs:
            return 0

        texts = [doc.page_content for doc in new_docs]
        embeddings = await asyncio.to_thread(self.embeddings.embed_documents, texts)
        metadatas = [_sanitize_metadata(doc.metadata) for doc in new_docs]

        await asyncio.to_thread(
            self._collection.add,
            ids=new_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(new_docs)

    async def search(
        self,
        query: str,
        top_k: int = None,
        where: Optional[dict] = None,
    ) -> list[Document]:
        """
        语义检索。

        Args:
            query: 查询文本
            top_k: 返回文档数
            where: ChromaDB 过滤条件，例如 {"folder": "1-自定义物品"}

        Returns:
            list[Document]: 按相似度降序排列的文档列表
        """
        top_k = top_k or RETRIEVAL_TOP_K
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)

        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=_restore_metadata(results["metadatas"][0][i] or {}),
            )
            doc.metadata["_score"] = 1 - results["distances"][0][i]
            documents.append(doc)

        return documents

    async def search_with_filter(
        self,
        query: str,
        folder: str = None,
        top_k: int = None,
    ) -> list[Document]:
        """
        带分类过滤的检索。

        可按文档所属文件夹过滤，适用于用户明确想查询特定分类的场景。
        """
        where = {"folder": folder} if folder else None
        return await self.search(query, top_k=top_k, where=where)

    async def reset(self) -> None:
        await asyncio.to_thread(self._client.delete_collection, self.collection_name)
        self._collection = await asyncio.to_thread(
            self._client.create_collection,
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def stats(self) -> dict:
        count = await asyncio.to_thread(self._collection.count)
        return {
            "集合名称": self.collection_name,
            "文档块数量": count,
            "存储路径": str(self.persist_dir),
            "嵌入模型": EMBEDDING_MODEL,
        }

    @property
    def client(self):
        """获取底层 ChromaDB 客户端（用于高级操作）。"""
        return self._client

    @property
    def collection(self):
        """获取底层 ChromaDB 集合。"""
        return self._collection
