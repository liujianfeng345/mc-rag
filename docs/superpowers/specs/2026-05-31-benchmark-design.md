# Benchmark 测试系统设计

## 目标

为 mc-rag 项目构建完整的 benchmark 测试体系，覆盖三个维度：

1. **版本对比** — 用同一套标准数据集一次性跑 v1/v2/v3/v4 四个版本，产出包含检索质量、生成质量、性能耗时的综合对比报告
2. **性能压测** — 测量端到端耗时和分段耗时（检索 / 生成 / 首 token），定位瓶颈
3. **回归检测** — 每次跑 benchmark 自动对比基线和上次跑分，标记退化/提升

## 架构

独立模块 `src/benchmark/`，调用现有 `src/eval/` 的评测逻辑，不重写评测本身。benchmark 的职责是：编排 + 计时 + 存储 + 对比。

```
src/benchmark/
├── __init__.py
├── cli.py               # CLI 入口，注册 benchmark 子命令
├── runner.py            # 多版本循环调度、生命周期管理
├── profiler.py          # 性能打点（阶段耗时 / 端到端 / 首 token）
├── comparator.py        # 基线对比、阈值检查、趋势标记
├── database.py          # SQLite CRUD、schema 管理、历史查询
├── reporter.py          # 终端 Rich 表格 + JSON 导出 + 趋势摘要
└── benchmark_config.py  # 默认版本列表、K 值、阈值、数据库路径
```

### 数据流

```
数据集 → Runner(循环 v1~v4)
  → 每个版本: retrieve → generate（Profiler 打点）
    → Profiler 记录各阶段耗时
    → 复用 src/eval/ 计算检索/生成指标
  → 汇总写入 SQLite
  → Comparator 对比基线，标记退化/提升
  → Reporter 输出终端报告 + JSON 文件
```

## 数据库设计

SQLite 文件位于 `benchmark_results/benchmark.db`。

### benchmark_runs — 每次 benchmark 记录

| 字段 | 类型 | 说明 |
|------|------|------|
| id | TEXT PK | 运行 ID，格式 `YYYYMMDD-HHMMSS` |
| timestamp | DATETIME | 执行时间 |
| agent_version | TEXT | v1/v2/v3/v4 |
| dataset_name | TEXT | 使用的数据集名称 |
| question_count | INTEGER | 问题总数 |
| git_commit | TEXT | 当前 HEAD commit hash |
| total_duration_ms | REAL | 总耗时（毫秒） |
| avg_ttft_ms | REAL | 平均首 token 延迟 |
| avg_retrieval_ms | REAL | 平均检索耗时 |
| avg_generation_ms | REAL | 平均生成耗时 |
| recall_at_5 | REAL | 检索 Recall@5 |
| precision_at_5 | REAL | 检索 Precision@5 |
| mrr | REAL | 检索 MRR |
| ndcg_at_5 | REAL | 检索 NDCG@5 |
| hit_at_5 | REAL | 检索 Hit@5 |
| faithfulness | REAL | RAGAS 忠实度 |
| answer_relevance | REAL | RAGAS 答案相关性 |
| context_relevance | REAL | RAGAS 上下文相关性 |
| passed | INTEGER | 是否通过阈值检查（0/1） |
| report_json | TEXT | 完整报告的 JSON 快照 |

### benchmark_baselines — 阈值配置

| 字段 | 类型 | 说明 |
|------|------|------|
| metric_name | TEXT PK | 指标名 |
| min_threshold | REAL | 最低阈值 |
| max_degradation_pct | REAL | 相比上次允许的最大退化百分比 |
| updated_at | DATETIME | 最后修改时间 |

## CLI 命令

```bash
# 跑全部版本（v1~v4），使用默认数据集
uv run python -m src.main benchmark

# 指定版本和数据集
uv run python -m src.main benchmark --versions v3,v4 -d eval_data/golden_50.json

# 仅性能压测（跳过质量评测）
uv run python -m src.main benchmark --profile-only

# 仅质量评测（跳过性能打点）
uv run python -m src.main benchmark --eval-only

# 打印历史趋势
uv run python -m src.main benchmark --history

# 指定版本查历史
uv run python -m src.main benchmark --history --versions v4

# 设置/更新基线阈值
uv run python -m src.main benchmark --set-baseline
```

## 配置

`src/benchmark/benchmark_config.py`：

```python
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4"]
DEFAULT_DATASET = "eval_data/golden_50.json"
RETRIEVAL_K_VALUES = [3, 5, 10]
PROFILE_STAGES = True
DB_PATH = "benchmark_results/benchmark.db"

DEFAULT_THRESHOLDS = {
    "recall_at_5":      {"min": 0.6, "max_degradation_pct": 15},
    "mrr":              {"min": 0.5, "max_degradation_pct": 15},
    "hit_at_5":         {"min": 0.7, "max_degradation_pct": 10},
    "faithfulness":     {"min": 0.6, "max_degradation_pct": 15},
    "answer_relevance": {"min": 0.6, "max_degradation_pct": 15},
}
```

## 报告输出示例

```
╔══════════════════════════════════════════════════╗
║     Benchmark Report 20260531-153042             ║
║     Dataset: golden_50.json | 50 questions       ║
╚══════════════════════════════════════════════════╝

  Metric            v1       v2       v3       v4
  ────────────────  ──────── ──────── ──────── ────────
  Recall@5          0.72     0.74     0.78     0.81 ↑
  MRR               0.65     0.67     0.70     0.73 ↑
  Faithfulness      0.71     0.73     0.80     0.85 ↑
  Avg TTFT (ms)     320      340      580      620
  Passed            ✓        ✓        ✓        ✓

  趋势: 本次 vs 上次
    v4 Recall@5: 0.81 (↑0.02)   Faithfulness: 0.85 (↑0.03)
    v3 Faithfulness: 0.80 (↓0.05 ⚠ 接近阈值)
```

## 模块职责

### profiler.py
- 通过 `time.perf_counter()` 在检索前、检索后、首 token 到达时、生成完成时打点
- 输出每条问题的耗时明细和版本汇总

### comparator.py
- 读取 `benchmark_baselines` 阈值
- 从 SQLite 查上一次同版本同数据集的跑分
- 判断：低于 `min_threshold` → 失败；相比上次下降超过 `max_degradation_pct` → 警告
- 设置 `passed` 字段

### database.py
- 初始化时自动建表
- 插入 `benchmark_runs` 记录
- 查询历史趋势（指定版本、时间范围）
- 读写 `benchmark_baselines`

### reporter.py
- 用 Rich 库输出终端彩色表格（版本对比表 + 趋势摘要）
- 可选导出完整 JSON 报告到 `benchmark_results/YYYYMMDD-HHMMSS.json`

## 数据集要求

分层混合策略，目标 50-60 题：

| 问题类型 | 数量 | 说明 |
|---------|------|------|
| 简单事实查询 | 15-20 | 如"什么是分帧" |
| 复杂推理 | 15-20 | 如"如何优化 Shader 提升低配置玩家体验" |
| 多步操作 | 15-20 | 如"如何将批量修改方块改为分帧处理" |

用 `dataset_builder.py` 批量生成初稿，人工审核后保存为 `eval_data/golden_50.json`。关键类别的题目需要标注 `relevant_sources`。

## 关键约束

- 不修改现有 `src/eval/` 模块，benchmark 作为消费者调用
- 复用现有 `src/main.py` 的版本路由逻辑，通过设置 `AGENT_VERSION` 环境变量切换
- SQLite 文件 gitignore，本地使用
- LLM 调用在评测中使用 temperature=0 保证可复现性
