# mc-rag

Minecraft 开发者文档 Agentic RAG 问答系统。

基于 LangGraph 构建有向图工作流，通过检索增强生成 (RAG) 技术，智能回答网易我的世界模组开发相关技术问题。

## 技术栈

- **Python** >= 3.11
- **LangChain** — LLM 调用与文档处理
- **LangGraph** — 有向图工作流编排
- **DeepSeek API** — 大语言模型（OpenAI 兼容接口）
- **ChromaDB** — 向量数据库（文档检索）
- **HuggingFace Embeddings** — 本地嵌入模型（多语言，支持中文）
- **Rich** — 终端交互界面

## 架构

项目提供两种 Agent 版本，可通过 `AGENT_VERSION` 环境变量切换（默认 v1）。

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

### 版本对比

| 特性 | v1（`src/agent/`） | v2（`src/agent_v2/`） |
|---|---|---|
| 路由方式 | 集中式：`grade_router` 函数 + `add_conditional_edges` | 分散式：节点内 `Command(goto=...)` |
| 图结构 | 静态边 + 条件边 | 仅 START → retrieve 一条边 |
| 可扩展性 | 新增节点需修改路由函数 | 新增节点只需在节点内指定跳转 |
| 职责划分 | graph 控制流，node 纯逻辑 | graph 极简，node 同时负责路由决策 |

### 工作流节点

| 节点 | 功能 |
|---|---|
| `retrieve` | 从 ChromaDB 检索相关文档 |
| `grade` | 用 LLM 逐篇评估文档相关性，过滤无关文档 |
| `generate` | 基于相关文档生成带引用来源的答案 |
| `rewrite` | 优化用户查询并重新检索（最多 2 次） |

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

# Agent 版本（v1: 静态路由, v2: 动态路由）
AGENT_VERSION=v1

# 文档配置
DOCS_DIR=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 向量数据库配置
VECTOR_DB_DIR=./chroma_db
```

### 使用

```bash
# 1. 构建文档向量索引（首次使用前必须执行）
uv run python -m src.main build

# 2. 单次问答（默认 v1 版本）
uv run python -m src.main ask "如何自定义物品？"

# 3. 使用 v2 动态路由版本
AGENT_VERSION=v2 uv run python -m src.main ask "如何自定义物品？"

# 4. 交互式问答
uv run python -m src.main demo
```

## 项目结构

```
mc-rag/
├── main.py                  # 根入口（占位）
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
│   ├── vector/
│   │   ├── vector_store.py       # ChromaDB 向量存储封装
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
