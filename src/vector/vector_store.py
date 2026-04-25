"""
向量存储模块

基于 ChromaDB 实现持久化向量存储，支持：
1. 文档嵌入与索引
2. 语义检索
3. 集合管理（重置、统计）
"""

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

    # -------------------------------------------------------------------------
    # 文档管理
    # -------------------------------------------------------------------------
    def add_documents(self, documents: list[Document]) -> int:
        """
        批量添加文档到向量存储。

        自动生成唯一 ID（基于 source + chunk_index），支持幂等添加。
        返回新增的文档数。
        """
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
        embeddings = self.embeddings.embed_documents(texts)
        metadatas = [doc.metadata for doc in new_docs]

        self._collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(new_docs)

    # -------------------------------------------------------------------------
    # 检索
    # -------------------------------------------------------------------------
    def search(
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
        query_embedding = self.embeddings.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            doc.metadata["_score"] = 1 - results["distances"][0][i]  # cosine → 相似度
            documents.append(doc)

        return documents

    def search_with_filter(
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
        return self.search(query, top_k=top_k, where=where)

    # -------------------------------------------------------------------------
    # 管理
    # -------------------------------------------------------------------------
    def reset(self) -> None:
        """清空向量存储并重建集合。"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> dict:
        """获取向量存储统计信息。"""
        count = self._collection.count()
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
