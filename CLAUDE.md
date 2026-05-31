# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 构建文档向量索引（首次使用前必须执行）
uv run python -m src.main build

# 单次问答
uv run python -m src.main ask "如何自定义物品？"

# 指定 Agent 版本
AGENT_VERSION=v4 uv run python -m src.main ask "如何自定义武器属性？"

# 交互式问答
uv run python -m src.main demo

# 启动 Streamlit Web 界面
uv run streamlit run src/web/app.py

# 启动 LangGraph API Server（自托管）
uv run langgraph dev

# RAG 评测
uv run python -m src.main eval -d eval_data/sample_questions.json                 # 完整评测
uv run python -m src.main eval -d eval_data/sample_questions.json --retrieval-only  # 仅检索评测
uv run python -m src.main eval -d eval_data/sample_questions.json --ragas-only -o report.json  # 仅生成质量评测并保存报告

# Benchmark 跨版本对比
uv run python -m src.main benchmark                     # v1~v4 完整对比
uv run python -m src.main benchmark --versions v1,v4    # 指定版本
uv run python -m src.main benchmark --profile-only      # 仅性能压测
uv run python -m src.main benchmark --history           # 查看历史趋势
uv run python -m src.main benchmark --set-baseline      # 设置基线阈值

# 评测数据集辅助生成（从知识库自动抽题）
uv run python -m src.eval.dataset_builder -c 10 -o eval_data/generated_questions.json
```

本项目没有测试套件和 lint 配置。

## 架构要点

### 四种 Agent 版本，同一套路由机制

`AGENT_VERSION` 环境变量（默认 v1）决定使用哪个 agent 模块。入口有两处，使用相同的版本分支模式：

- **CLI** (`src/main.py:37-44`)：动态 `import` 对应版本的 `build_rag_graph`
- **Web API** (`src/web/api.py:28-38`)：同理，按版本字符串构建图

每个版本是自包含模块（`src/agent/`, `src/agent_v2/`, `src/agent_v3/`, `src/agent_v4/`），各有独立的 `graph.py` / `node.py` / `prompt.py` / `state.py`。

### 两套 State 类型，不能混用

- **v1 / v2** 使用 `RAGState`（`src/agent/state.py`），包含 `generation`、`current_query`、`rewrite_count` 字段
- **v3 / v4** 使用 `AgentState`（`src/agent_v3/state.py` 和 `src/agent_v4/state.py`），包含 `final_report`、`findings`、`research_plan` 字段。v4 的 `AgentState` 额外增加了 `reflection`、`refine_count`、`supplemental_findings`

代码中判断"是否为旧版 state"的条件是 `AGENT_VERSION in ("v1", "v2")`——处理 input_state 初始化（v1/v2 需要 `rewrite_count`）和输出字段名（`generation` vs `final_report`）时都要注意这个分支。

### 流式输出只捕获最终答案节点的 token

`src/web/api.py:62-63` 中 `OUTPUT_NODES` 字典按版本指定了答案生成节点名（v1/v2 → `generate`，v3/v4 → `synthesize`）。流式事件循环中按 `langgraph_node` metadata 过滤，只让最终答案节点的 token 流向用户，避免中间节点（研究员 ReAct 循环、反思评估等）的 LLM 输出混入展示。

### 文档来源的存储策略

文档来源信息被拼入消息内容的 markdown（`src/web/app.py:217-222`），而非使用独立的 `st.divider` / `st.caption` 展示。这是因为 `st.chat_message` 容器在 Streamlit 跨渲染周期会错位——将来源作为消息内容的一部分持久存储在 `session_state.messages` 中解决了这一问题。

### v3 → v4 的精炼闭环

v4 是 v3 的超集，新增 `reflect` 和 `refine` 两个节点，其余节点代码几乎相同。`reflect_node` 用结构化输出（Pydantic `ReflectionResult`）评估答案质量，`refine_node` 对 `follow_up_queries` 并行执行 `hybrid_search` 补充检索。`_route_after_reflect` 处理条件路由：答案充分 → END，有缺口且未超上限 → refine → synthesize 循环，达到 `MAX_REFINE_ITERATIONS`（默认 2）→ 降级 END。`state.get("reflection", {}).get("is_sufficient", True)` 的默认值是 `True`——state 中 reflection 缺失时安全退出。

### 混合检索与 RRF 重排序

`VectorStore.hybrid_search()` 同时执行语义检索（ChromaDB 余弦相似度）和 BM25 关键词检索，通过 RRF（Reciprocal Rank Fusion，k=60）合并排序。RRF 的核心逻辑在 `src/vector/vector_store.py:247-297`。BM25 检索器从 ChromaDB 加载全部文档初始化，延迟构建（首次 hybrid_search 时触发）。

### 文档加载的幂等性

`VectorStore.add_documents()` 按 `source:chunk_index` 格式生成文档 ID，已存在的 ID 会被跳过。这意味着多次运行 `build` 命令不会导致重复索引。

### ChromaDB metadata 序列化

ChromaDB 不接受 dict/list 类型的 metadata 值。`_sanitize_metadata` / `_restore_metadata` 工具函数处理 JSON 序列化/反序列化——写入时将 dict/list 转为 JSON 字符串，读取时尝试还原。这主要影响 RRF score 和原始 metadata 中可能的嵌套结构。

### LangSmith 追踪

在 `src/main.py:26-34` 中按 `LANGCHAIN_TRACING_V2` 环境变量条件初始化。LangSmith 回放时 state 中可能没有 `question` 字段——`_get_question()` 函数（v3/v4 node.py 中）兼容从 `messages` 列表末尾提取问题文本。

### 评测模块 (`src/eval/`)

提供三层评测能力：

| 模块 | 功能 | 指标 |
|------|------|------|
| `retrieval.py` | 检索质量 | Recall@K, Precision@K, MRR, NDCG@K, Hit@K |
| `ragas_eval.py` | 生成质量（LLM 裁判） | Faithfulness, Answer Relevance, Context Relevance |
| `runner.py` | 统一入口 | 整合以上两种评测，输出综合报告 |

数据集格式（`eval_data/*.json`）：JSON 数组，每项包含 `question`（必填）、`relevant_sources`（检索评测用）、`golden_answer`（展示用）。`dataset_builder.py` 可从知识库随机采样文档片段，用 LLM 自动生成候选问题，加速数据集构建。

检索评测依赖 `relevant_sources` 标注——检索结果的文件路径与标注的 `relevant_sources` 做前缀匹配来判断相关性。RAGAS 评测不依赖标注，但需要调用 LLM（评测用 temperature=0 保证一致性）。
