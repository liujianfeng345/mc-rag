# 评测难度分层设计

## 目标

为评测数据集的每道题自动标注难度（简单/中等/复杂），在 eval 和 benchmark 的输出中按难度分组展示指标，让系统能力的强弱项一目了然。

## 动机

- 总平均分掩盖了系统在不同难度上的表现差异
- 一个系统可能简单题满分、复杂题不及格，但平均分看起来还不错
- 按难度分层后，面试/汇报时能更有说服力地展示系统能力边界
- 跨版本对比时，分层数据能揭示"新版本到底在哪类问题上变好了"

## 难度分类

### 分类方式

LLM 自动分类，运行时对每道题调用一次分类 LLM（temperature=0），分类标准聚焦问题本身的推理深度和知识点数量：

| 难度 | 标准 | 示例 |
|------|------|------|
| 简单 | 单一知识点，直接查询即可回答 | "如何注册一个自定义基础物品？" |
| 中等 | 2-3 个知识点，需要一定推理或跨段落整合 | "自定义盔甲与基础物品相比多了哪些步骤？" |
| 复杂 | 多知识点、多步推理、涉及计算或跨文档整合 | "铁砧结合附魔物品的经验消耗如何计算？" |

### 分类时机

数据集加载后、评测开始前。`EvalDataset` 新增 `classify_difficulty()` 方法，批量分类后将结果写入 `EvalItem.difficulty`。

分类失败时，`difficulty` 保持空字符串 `""`，在分组统计中归入"未分类"。

## 数据模型变更

### EvalItem（`src/eval/dataset.py`）

新增字段：

```python
difficulty: str = ""  # "简单" | "中等" | "复杂" | ""
```

### EvalDataset（`src/eval/dataset.py`）

新增方法：

```python
async def classify_difficulty(self) -> None:
    """对数据集中所有题目进行难度分类，结果写入 EvalItem.difficulty"""
```

- 使用 `DEEPSEEK_API_KEY` 的 LLM，temperature=0
- 并发调用以加速分类
- 每条题一次 LLM 调用，Prompt 内置分类标准
- 异常不中断，单题失败不影响其他题
- 新增 `difficulty_distribution` 属性返回各难度题目数

## 评测结果分组

### EvalResult（`src/eval/retrieval.py`）

新增属性：

```python
@property
def by_difficulty(self) -> dict[str, "EvalResult"]:
    """按难度分组返回各自的 EvalResult"""
```

内部逻辑：将 `self.metrics` 按 `difficulty` 分组，每组构造一个新的 `EvalResult`，复用现有的 avg_recall / avg_mrr 等聚合属性和 rich_table 输出。

### RAGASResult（`src/eval/ragas_eval.py`）

新增属性：

```python
@property
def by_difficulty(self) -> dict[str, "RAGASResult"]:
    """按难度分组返回各自的 RAGASResult"""
```

同样分组聚合 avg_faithfulness / avg_answer_relevance / avg_context_relevance。

### 终端输出（`src/eval/runner.py`）

`FullEvalReport.print()` 在原有总览表之后，追加难度分组表。示例：

```
检索评测: golden_50
┌──────────┬────────┬───────────┬────────┬──────┐
│ 难度     │  题目数 │ Recall@5  │ MRR    │ Hit@5│
├──────────┼────────┼───────────┼────────┼──────┤
│ 简单     │     18 │    85.2%  │ 0.720  │ 94.4%│
│ 中等     │     22 │    72.1%  │ 0.583  │ 81.8%│
│ 复杂     │     15 │    54.7%  │ 0.401  │ 60.0%│
├──────────┼────────┼───────────┼────────┼──────┤
│ 总计     │     55 │    71.8%  │ 0.576  │ 80.0%│
└──────────┴────────┴───────────┴────────┴──────┘
```

## Benchmark 改造

### VersionResult（`src/benchmark/runner.py`）

新增字段：

```python
ragas_by_difficulty: dict[str, dict] = field(default_factory=dict)
# {"简单": {"faithfulness": 0.85, "answer_relevance": 0.82, "context_relevance": 0.73}, ...}
```

### Reporter（`src/benchmark/reporter.py`）

版本对比表按难度展开，每个难度块显示各版本的检索指标和 RAGAS 指标：

```
版本对比 (按难度分组)
                          V1      V2      V3      V4
────────────────────────────────────────────────────
[简单 18题]
  Faithfulness          85.2%   86.1%   87.3%   89.0%
  Answer Relevance      82.1%   83.5%   84.2%   86.7%
  Context Relevance     75.3%   76.0%   77.1%   78.5%
[中等 22题]
  Faithfulness          78.3%   80.2%   82.5%   84.1%
  ...
[复杂 15题]
  Faithfulness          62.5%   65.8%   70.2%   73.6%
  ...
```

### Comparator（`src/benchmark/comparator.py`）

基线检测增加对各难度分层指标的检查：任意难度的任意 RAGAS 指标低于阈值均返回不通过。

### Database（`src/benchmark/database.py`）

`benchmark_runs` 表新增 12 列：每个难度（简单/中等/复杂）× 3 个 RAGAS 指标 + 1 个检索指标（Recall@5）。

## 改动文件清单

| 文件 | 改动 |
|------|------|
| `src/eval/dataset.py` | EvalItem.difficulty + EvalDataset.classify_difficulty() |
| `src/eval/retrieval.py` | RetrievalMetrics.difficulty + EvalResult.by_difficulty |
| `src/eval/ragas_eval.py` | RAGASMetrics.difficulty + RAGASResult.by_difficulty |
| `src/eval/runner.py` | 调用 classify + 输出分组表 + to_dict 含分组 |
| `src/benchmark/runner.py` | VersionResult 含难度分层 RAGAS + _compute_ragas_batch 返回分组 |
| `src/benchmark/reporter.py` | 版本对比表按难度展开 |
| `src/benchmark/comparator.py` | 基线检测含分层指标 |
| `src/benchmark/database.py` | 表结构加难度分层列 |
| `src/benchmark/benchmark_config.py` | 阈值可不变（或按难度设不同阈值） |

## 不做什么

- 不修改已有 json 数据集文件
- 不改动检索/RAGAS 的核心计算逻辑
- 不改变现有 eval/benchmark 的命令行接口
- 分类失败时静默降级，不阻塞评测流程
