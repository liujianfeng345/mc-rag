# mc-rag

Minecraft 开发者文档 Agentic RAG 问答系统。

基于 LangGraph 构建有向图工作流，通过检索增强生成 (RAG) 技术，智能回答网易我的世界模组开发相关技术问题。

## 技术栈

- **Python** >= 3.11
- **LangChain** — LLM 调用与文档处理
- **LangGraph** — 有向图工作流编排
- **DeepSeek API** — 大语言模型（OpenAI 兼容接口）
- **ChromaDB** — 向量数据库（文档检索）

## 架构

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

# 激活虚拟环境
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # Linux/macOS
```

### 配置

复制环境变量模板并填入 DeepSeek API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
RETRIEVAL_TOP_K=5
```

### 运行

```bash
python main.py
```

## 项目结构

```
mc-rag/
├── main.py                 # 入口文件
├── pyproject.toml          # 项目元数据与依赖
├── .env                    # 环境变量（不提交）
├── .env.example            # 环境变量模板
├── src/
│   ├── agent/
│   │   ├── graph.py        # LangGraph 图构建
│   │   ├── node.py         # 各节点函数实现
│   │   ├── prompt.py       # LLM 提示词模板
│   │   └── state.py        # 图状态类型定义
│   └── utils/
│       └── config.py       # 配置管理
└── data/
    └── mcguide/            # 我的世界开发者指南文档
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
