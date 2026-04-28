"""mc-rag Streamlit Web 应用 — Minecraft 开发文档 RAG 问答助手

启动方式：
    uv run streamlit run src/web/app.py
"""

import streamlit as st
import asyncio
from pathlib import Path

# 页面配置 — 必须是第一个 st 调用
st.set_page_config(
    page_title="Minecraft RAG 助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dotenv import load_dotenv

load_dotenv()

from src.vector.vector_store import VectorStore
from src.web.api import stream_rag_response, validate_input
from src.utils.config import DOCS_DIR, RETRIEVAL_TOP_K, LLM_MODEL, EMBEDDING_MODEL

# =============================================================================
# 缓存资源（避免每次渲染重新加载嵌入模型和向量库）
# =============================================================================
@st.cache_resource(show_spinner="正在加载向量库和嵌入模型...")
def get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_data(show_spinner=False)
def get_store_stats() -> dict:
    """获取向量库统计信息（同步包装）"""
    store = get_vector_store()
    return asyncio.run(store.stats())


# =============================================================================
# 会话状态初始化
# =============================================================================
def init_session():
    """初始化 session_state 默认值"""
    defaults = {
        "messages": [],       # 对话历史 [{"role": "user/assistant", "content": "..."}]
        "history": [],        # LangChain 消息对象列表（传给图）
        "agent_version": "v4",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# 侧边栏
# =============================================================================
def render_sidebar():
    """渲染侧边栏：版本选择、系统统计、操作按钮、技术说明入口"""
    with st.sidebar:
        st.title("📚 MC-RAG 助手")

        # Agent 版本选择
        st.subheader("🔧 Agent 版本")
        version_labels = {
            "v4": "v4 · 自我反思精炼（推荐）",
            "v3": "v3 · 问题分解+并行研究",
            "v2": "v2 · 动态路由",
            "v1": "v1 · 静态路由",
        }
        selected = st.selectbox(
            "选择 RAG 策略",
            options=["v4", "v3", "v2", "v1"],
            format_func=lambda v: version_labels[v],
            index=["v4", "v3", "v2", "v1"].index(st.session_state.agent_version),
            help="v4 最适合复杂问题，v1 适合简单检索",
        )
        if selected != st.session_state.agent_version:
            st.session_state.agent_version = selected
            # 切换版本时不保留旧对话（状态结构不同）
            st.session_state.messages = []
            st.session_state.history = []

        st.divider()

        # 系统状态
        st.subheader("📊 系统状态")
        try:
            stats = get_store_stats()
            st.metric("文档块数", stats.get("文档块数量", "N/A"))
            st.metric("集合名称", stats.get("集合名称", "N/A"))
            st.caption(f"检索 Top-K: {RETRIEVAL_TOP_K}")
            st.caption(f"LLM: {LLM_MODEL}")
            emb_name = EMBEDDING_MODEL.split("/")[-1] if "/" in EMBEDDING_MODEL else EMBEDDING_MODEL
            st.caption(f"嵌入: {emb_name}")
        except Exception as e:
            st.warning(f"向量库暂不可用: {e}")

        st.divider()

        # 操作按钮
        if st.button("🗑 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        st.divider()

        # 技术说明
        with st.expander("📖 技术原理"):
            render_tech_info()

        # 页脚
        st.caption(f"知识库路径: {DOCS_DIR}")
        st.caption("Powered by LangGraph + DeepSeek")


# =============================================================================
# 技术原理说明
# =============================================================================
def render_tech_info():
    """展示 RAG 系统架构、版本对比等技术说明"""
    st.markdown("### 系统架构")
    st.markdown(
        """
        本系统基于 **Agentic RAG** 范式，使用 LangGraph 编排多节点有向图工作流，
        结合 DeepSeek 大模型与 ChromaDB 向量数据库实现智能问答。

        **核心流程：**
        1. **检索 (Retrieve)** — 从 Minecraft 开发文档库中通过语义 + 关键词混合检索相关文档
        2. **评估 (Grade)** — LLM 评估检索结果的相关性
        3. **生成 (Generate)** — 基于相关文档生成答案
        4. **重写/反思 (Rewrite/Reflect)** — 答案不足时自动优化查询或补充检索
        """
    )

    st.markdown("### 四个 Agent 版本")

    versions_info = [
        ("v1 · 静态路由", "`src/agent/`",
         "路由在构图时静态定义。检索 → 评估 → 生成/重写。适合简单问答。"),
        ("v2 · 动态路由", "`src/agent_v2/`",
         "节点通过 Command 动态指定下一跳。更灵活的控制流。"),
        ("v3 · 深度研究", "`src/agent_v3/`",
         "问题分解为子问题 → 并行 ReAct 研究员 → 综合生成。适合复杂问题。"),
        ("v4 · 自我反思", "`src/agent_v4/`",
         "在 v3 基础上增加 reflect + refine 节点，形成精炼闭环。最多迭代 2 轮。**推荐**。"),
    ]

    for title, path, desc in versions_info:
        st.markdown(f"**{title}** ({path})：{desc}")

    st.markdown("### 技术栈")
    st.markdown(
        """
        | 组件 | 技术 |
        |------|------|
        | 图编排 | LangGraph |
        | 大模型 | DeepSeek Chat API |
        | 向量库 | ChromaDB |
        | 嵌入模型 | BGE-small-zh-v1.5 (本地) |
        | 混合检索 | 语义 + BM25 + RRF 重排序 |
        | Web 框架 | Streamlit |
        """
    )


# =============================================================================
# 主对话区
# =============================================================================
def render_chat():
    """渲染对话界面并处理用户输入"""
    vector_store = get_vector_store()

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 欢迎消息
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "你好！我是 **Minecraft 开发文档 RAG 助手**。\n\n"
                "我可以回答关于网易我的世界模组开发的技术问题，所有答案基于官方开发文档。\n\n"
                "试着问一个问题吧，比如「如何注册一个物品？」或「事件系统怎么用？」"
            )

    # 输入框
    if prompt := st.chat_input("输入你的问题...", max_chars=500):
        # 验证输入
        error = validate_input(prompt)
        if error:
            st.toast(error, icon="⚠️")
            return

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 RAG 并流式展示
        with st.chat_message("assistant"):
            try:
                # 用可变容器收集 done 数据（async generator 不支持 return value）
                done: dict = {}
                st.write_stream(_stream_tokens(prompt, vector_store, done))

                if done.get("type") == "done":
                    answer = done.get("answer", "")
                    sources = done.get("sources", [])

                    # 将来源文档拼入消息内容，作为 markdown 持久存储
                    # 避免单独 st.divider/st.caption 在 st.chat_message 容器中跨渲染周期错位
                    content = answer
                    if sources:
                        lines = ["\n\n---\n📄 **参考文档来源:**"]
                        for s in sources:
                            lines.append(f"- {s['source']} (相关度: {s['score']:.3f})")
                        content += "\n".join(lines)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )
                    if done.get("history_msg"):
                        st.session_state.history.append(done["history_msg"])
            except Exception as e:
                st.error(f"生成回答时出错: {e}")


async def _stream_tokens(prompt: str, vector_store: VectorStore, done: dict):
    """异步生成器，yield 字符串 token 供 st.write_stream 展示，
    将完成信号写入 done 容器（Python async generator 不能用 return 传值）。"""
    agent_version = st.session_state.agent_version
    history = st.session_state.history

    async for chunk in stream_rag_response(
        question=prompt,
        vector_store=vector_store,
        history=history,
        agent_version=agent_version,
    ):
        if chunk["type"] == "token":
            yield chunk["content"]
        elif chunk["type"] == "done":
            done.update(chunk)


# =============================================================================
# 主入口
# =============================================================================
def main():
    init_session()
    render_sidebar()

    # 主区域标题
    st.title("Minecraft 开发文档 RAG 问答助手")
    st.caption("基于 LangGraph Agentic RAG 的智能技术问答系统 · 所有答案来源于官方开发文档")

    st.divider()
    render_chat()


if __name__ == "__main__":
    main()
