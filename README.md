# mc-rag

Minecraft 开发者文档 Agentic RAG 问答系统。

基于 LangGraph 构建有向图工作流，通过检索增强生成 (RAG) 技术，智能回答网易我的世界模组开发相关技术问题。

## 技术栈

- **Python** >= 3.11
- **LangChain** — LLM 调用与文档处理
- **LangGraph** — 有向图工作流编排
- **DeepSeek API** — 大语言模型（OpenAI 兼容接口）
- **ChromaDB** — 向量数据库（语义检索）
- **BM25 + RRF** — 关键词检索与重排序（混合检索，提升召回率）
- **HuggingFace Embeddings** — 本地嵌入模型（多语言，支持中文）
- **LangSmith** — LLM 可观测性追踪
- **Rich** — 终端交互界面

## 架构

项目提供三种 Agent 版本，可通过 `AGENT_VERSION` 环境变量切换（默认 v1）。

### v1（静态路由）— `src/agent/`

路由逻辑集中在 `graph.py` 的 `grade_router` 函数中，通过 `add_conditional_edges` 在构建图时静态定义。

```
                    ┌─────────────────────────────────────┐
                    │              START                  │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │      retrieve       │  从向量库检索相关文档
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │       grade         │  LLM 评估文档相关性
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │ 文档≥2 篇     │              │ 文档不足
                │              │ 重写<2次      │
       ┌────────▼────────┐    │    ┌──────────▼──────────┐
       │    generate     │    │    │      rewrite        │  优化查询
       │  生成最终答案    │    │    └──────────┬──────────┘
       └────────┬────────┘    │               │
                │             │ 重写≥2次       │
                │             │ 仍无文档       │
                │    ┌────────▼──────────┐    │
                │    │    generate       │    │  降级生成
                │    │  （降级回答）      │    │
                │    └────────┬──────────┘    │
                │             │               │
                └──────┬──────┘───────────────┘
                       │
                ┌──────▼──────┐
                │     END     │
                └─────────────┘
```

### v2（动态路由）— `src/agent_v2/`

路由逻辑分散在各个节点内部，每个节点通过 `Command(goto=...)` 动态指定下一跳，`graph.py` 只定义 START → retrieve 一条边，不再需要集中的路由函数。

```
                    ┌─────────────────────────────────────┐
                    │              START                  │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │      retrieve       │  ──→ grade（Command 指定）
                    └──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │       grade         │  ──→ generate / rewrite（Command 指定）
                    └──────────────────────┘
                      ↙                ↘
            ┌──────────▼────┐    ┌──────▼──────────┐
            │   generate    │    │    rewrite      │  ──→ retrieve / generate
            │  (Command 到  │    │  (Command 指定)  │      （重写≥2次则直接生成）
            │     END)      │    └──────────────────┘
            └───────┬───────┘
                    │
                    ▼
                    END
```

### v3（深度研究）— `src/agent_v3/`

采用问题分解 + 并行 ReAct 研究员架构。先将用户问题分解为 2-4 个子问题，每个子问题由独立研究员在 ReAct 循环中自主进行多轮检索，最终汇总所有研究发现生成结构化回答。

```
                    ┌──────────────────────────────────────┐
                    │               START                  │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     decompose       │  LLM 分解问题 → 子问题列表
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   research_batch    │  并行 ReAct 研究员
                    │  ┌────────────────┐ │
                    │  │ 研究员1 (ReAct) │ │  多轮检索 + 思考
                    │  │ 研究员2 (ReAct) │ │
                    │  │ 研究员3 (ReAct) │ │
                    │  └────────────────┘ │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    synthesize       │  汇总研究发现，生成结构化答案
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │        END          │
                    └─────────────────────┘
```

### 版本对比

| 特性 | v1（`src/agent/`） | v2（`src/agent_v2/`） | v3（`src/agent_v3/`） |
|---|---|---|---|---|
| 路由方式 | 集中式：`grade_router` + 条件边 | 分散式：节点内 `Command(goto=...)` | 线性管道：固定顺序，无分支 |
| 图结构 | 静态边 + 条件边 | 仅 START → retrieve 一条边 | START → decompose → research_batch → synthesize → END |
| 检索策略 | 单次混合检索 + LLM 评分过滤 | 单次混合检索 + LLM 评分过滤 | 多轮、多角度 ReAct 自主检索，无需显式评分 |
| 查询重写 | 图级别循环重写（最多 2 次） | 图级别循环重写（最多 2 次） | 问题分解替代重写，研究员自主调整检索角度 |
| 并行能力 | 文档评分并行 | 文档评分并行 | 子问题级别并行研究 |
| 适用场景 | 简单事实查询 | 简单事实查询 | 复杂、多方面的技术问题 |

### 工作流节点

#### v1 / v2 共用节点

| 节点 | 功能 |
|---|---|
| `retrieve` | 混合检索：语义检索 + BM25 关键词检索 + RRF 重排序 |
| `grade` | 用 LLM 逐篇评估文档相关性，过滤无关文档 |
| `generate` | 基于相关文档生成带引用来源的答案 |
| `rewrite` | 优化用户查询并重新检索（最多 2 次） |

#### v3 专属节点

| 节点 | 功能 |
|---|---|
| `decompose` | LLM 将用户问题分解为 2-4 个独立子问题 |
| `research_batch` | 并行启动多个 ReAct 研究员，每个研究员自主进行多轮检索与思考 |
| `synthesize` | 汇总所有研究发现，生成结构化、带来源引用的最终回答 |

## 快速开始

### 前置要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd mc-rag

# 安装依赖
uv sync
```

### 配置

复制环境变量模板并填入 DeepSeek API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# DeepSeek API 配置（兼容 OpenAI SDK）
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 模型配置
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Agent 版本（v1: 静态路由, v2: 动态路由, v3: 深度研究）
AGENT_VERSION=v1

# 文档配置
DOCS_DIR=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 向量数据库配置
VECTOR_DB_DIR=./chroma_db

# 混合检索参数（可选）
BM25_TOP_K=5
RRF_K=60

# LangSmith 追踪配置（可选，默认关闭）
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=mc-rag
```

### 使用

```bash
# 1. 构建文档向量索引（首次使用前必须执行）
uv run python -m src.main build

# 2. 单次问答（默认 v1 版本）
uv run python -m src.main ask "如何自定义物品？"

# 3. 使用 v2 动态路由版本
AGENT_VERSION=v2 uv run python -m src.main ask "如何自定义物品？"

# 4. 使用 v3 深度研究版本（适合复杂问题）
AGENT_VERSION=v3 uv run python -m src.main ask "如何创建一个自定义生物并为其添加攻击技能？"

# 5. 交互式问答
uv run python -m src.main demo

# 6. 启动 LangGraph API Server（LangSmith 平台 / 自托管）
uv run langgraph dev
```

## 项目结构

```
mc-rag/
├── main.py                  # 根入口（占位）
├── graph_server.py          # LangGraph API Server 图加载入口
├── langgraph.json           # LangGraph Server 配置文件
├── pyproject.toml           # 项目元数据与依赖
├── .env                     # 环境变量（不提交）
├── .env.example             # 环境变量模板
├── src/
│   ├── main.py              # CLI 入口（build / ask / demo）
│   ├── agent/                    # v1: 静态路由版本
│   │   ├── graph.py              # LangGraph 图构建与路由决策
│   │   ├── node.py               # 各节点函数实现（检索/评分/生成/重写）
│   │   ├── prompt.py             # LLM 提示词模板
│   │   └── state.py              # 图状态类型定义
│   ├── agent_v2/                 # v2: 动态路由版本（Command 模式）
│   │   ├── graph.py              # LangGraph 图构建（仅 START → retrieve）
│   │   ├── node.py               # 节点函数 + 路由决策（Command(goto=...)）
│   │   ├── prompt.py             # LLM 提示词模板
│   │   └── state.py              # 图状态类型定义
│   ├── agent_v3/                 # v3: 深度研究版本（问题分解 + ReAct 研究员）
│   │   ├── graph.py              # LangGraph 图构建（线性管道）
│   │   ├── node.py               # 节点函数：decompose / research_batch / synthesize
│   │   ├── prompt.py             # LLM 提示词模板
│   │   └── state.py              # 图状态类型定义
│   ├── vector/
│   │   ├── vector_store.py       # ChromaDB 向量存储封装（支持语义 + BM25 混合检索）
│   │   └── document_loader.py    # Markdown 文档加载与分块
│   └── utils/
│       └── config.py        # 配置管理（环境变量读取）
├── data/
│   └── mcguide/             # 我的世界开发者指南文档
├── chroma_db/               # 向量数据库持久化目录（自动生成）
└── image/                   # 项目截图
```

## 知识源

文档来自**网易我的世界开发者指南**，涵盖：

- 自定义物品（基础物品、武器、盔甲、3D 物品等）
- 自定义方块（JSON 组件、特殊方块、方块实体等）
- 自定义生物、维度、配方、生物群系
- 自定义附魔、状态效果、音乐、指令
- 自定义成就系统、书本、远程武器等

## 许可证

MIT
