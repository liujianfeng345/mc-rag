# 评测难度分层 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为评测数据集的每道题自动标注难度（简单/中等/复杂），在 eval 和 benchmark 输出中按难度分组展示检索+RAGAS 指标。

**Architecture:** 数据集加载后调用 LLM 批量分类难度 → 写入 EvalItem.difficulty → 评测各环节携带 difficulty 字段 → 结果层按 difficulty 分组聚合，复用现有平均值计算和表格输出。不改动检索/RAGAS 核心算法。

**Tech Stack:** Python, dataclass, asyncio, DeepSeek Chat API, Rich 终端表格, SQLite

---

### Task 1: EvalItem 与 EvalDataset —— 难度分类

**Files:**
- Modify: `src/eval/dataset.py`

- [ ] **Step 1: EvalItem 新增 difficulty 字段**

在 `EvalItem` dataclass 中加字段（第 25-30 行附近）：

```python
@dataclass
class EvalItem:
    """单条评测数据。"""
    question: str
    relevant_sources: list[str] = field(default_factory=list)
    golden_answer: str = ""
    difficulty: str = ""  # "简单" | "中等" | "复杂" | ""
```

- [ ] **Step 2: EvalDataset 新增 classify_difficulty 方法和分类 Prompt**

在 `EvalDataset` 类末尾（第 77 行 `stats()` 方法之后）新增：

```python
    async def classify_difficulty(self) -> None:
        """对数据集中所有题目进行 LLM 难度分类，结果写入 EvalItem.difficulty。"""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from ..utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

        PROMPT = """你的任务是判断一个"RAG 系统的用户问题"的难度等级。

判断标准（只看问题本身的推理深度和涉及的知识点数量）：
- 简单：单一知识点，直接查询文档即可回答，不需要推理
- 中等：涉及 2-3 个知识点，需要一定推理或跨段落信息整合
- 复杂：涉及多个知识点、需要多步推理、涉及计算或需要跨文档整合信息

问题：{question}

请只输出一个词：简单、中等 或 复杂。"""

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
            max_tokens=32,
        )

        async def _classify_one(item: EvalItem) -> None:
            try:
                msg = await llm.ainvoke([
                    HumanMessage(content=PROMPT.format(question=item.question))
                ])
                text = msg.content.strip()
                if "复杂" in text:
                    item.difficulty = "复杂"
                elif "中等" in text:
                    item.difficulty = "中等"
                elif "简单" in text:
                    item.difficulty = "简单"
            except Exception:
                pass  # 分类失败保持 ""

        await asyncio.gather(*[_classify_one(item) for item in self.items])

    @property
    def difficulty_distribution(self) -> dict[str, int]:
        """返回各难度题目数量分布。"""
        dist: dict[str, int] = {}
        for item in self.items:
            key = item.difficulty or "未分类"
            dist[key] = dist.get(key, 0) + 1
        return dist
```

注意：文件顶部需加 `import asyncio`。

- [ ] **Step 3: 验证 dataset 模块可正常导入**

```bash
cd C:\Users\87362\Desktop\agent\mc-rag && uv run python -c "from src.eval.dataset import EvalDataset, EvalItem; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/eval/dataset.py
git commit -m "feat(eval): EvalItem 新增 difficulty 字段 + EvalDataset.classify_difficulty()"
```

---

### Task 2: RetrievalMetrics 携带 difficulty + EvalResult 分组

**Files:**
- Modify: `src/eval/retrieval.py`

- [ ] **Step 1: RetrievalMetrics 新增 difficulty 字段**

在第 67 行 `question` 字段后加：

```python
@dataclass
class RetrievalMetrics:
    """单次检索的指标。"""
    question: str
    difficulty: str = ""  # 来自 EvalItem.difficulty
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_at_k: dict[int, bool] = field(default_factory=dict)
    retrieved_sources: list[str] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)
```

- [ ] **Step 2: _compute_metrics 传入并写入 difficulty**

修改 `_compute_metrics` 签名（第 222 行），增加 `difficulty` 参数：

```python
    def _compute_metrics(
        self,
        item: EvalItem,
        retrieved_sources: list[str],
        k_values: list[int],
    ) -> RetrievalMetrics:
        # ... 现有计算逻辑不变 ...
        return RetrievalMetrics(
            question=item.question,
            difficulty=item.difficulty,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_at_k=hit_at_k,
            retrieved_sources=retrieved_sources[:max(k_values)],
            relevant_sources=list(relevant_set),
        )
```

- [ ] **Step 3: _compute_metrics_auto 同样传入 difficulty**

修改 `_compute_metrics_auto` 签名（第 386 行），增加 `difficulty` 参数，并在返回 `RetrievalMetrics` 时写入。

`evaluate_auto` 调用处传入 `item.difficulty`。

- [ ] **Step 4: EvalResult 新增 by_difficulty 属性**

在 `EvalResult` 类中（第 125 行 `summary()` 之后）新增：

```python
    @property
    def by_difficulty(self) -> dict[str, "EvalResult"]:
        """按难度分组返回各自的 EvalResult。键为 "简单"/"中等"/"复杂"/"未分类"。"""
        groups: dict[str, list[RetrievalMetrics]] = {"简单": [], "中等": [], "复杂": [], "未分类": []}
        for m in self.metrics:
            key = m.difficulty or "未分类"
            groups.setdefault(key, []).append(m)
        result: dict[str, "EvalResult"] = {}
        for diff, metrics_list in groups.items():
            if not metrics_list:
                continue
            result[diff] = EvalResult(
                dataset_name=f"{self.dataset_name} ({diff})",
                metrics=metrics_list,
                k_values=self.k_values,
            )
        return result
```

- [ ] **Step 5: Commit**

```bash
git add src/eval/retrieval.py
git commit -m "feat(eval): RetrievalMetrics 携带 difficulty + EvalResult 按难度分组"
```

---

### Task 3: RAGASMetrics 携带 difficulty + RAGASResult 分组

**Files:**
- Modify: `src/eval/ragas_eval.py`

- [ ] **Step 1: RAGASMetrics 新增 difficulty 字段**

在第 117 行 `question` 字段后加：

```python
@dataclass
class RAGASMetrics:
    """单条问答的生成质量指标。"""
    question: str
    difficulty: str = ""  # 来自 EvalItem.difficulty
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    answer: str = ""
    error: str = ""
```

- [ ] **Step 2: _evaluate_single 接收 item 并写入 difficulty**

修改 `_evaluate_single`（第 259 行）签名，参数从 `item` 改为显式传入所需字段；或者直接把 `item.difficulty` 在调用处传入后在 `RAGASMetrics` 的 `difficulty=` 写入。最简方式——在 `evaluate()` 中调用 `_evaluate_single` 后补写 difficulty：

```python
    async def evaluate(
        self,
        dataset: EvalDataset,
        verbose: bool = True,
    ) -> RAGASResult:
        # ... 循环内改为 ...
        for idx, item in enumerate(dataset.items):
            if console:
                console.print(f"[dim][{idx + 1}/{len(dataset)}] 评测: {item.question[:80]}...[/dim]")
            try:
                metrics = await self._evaluate_single(item)
                metrics.difficulty = item.difficulty  # 补写难度
                all_metrics.append(metrics)
            except Exception as e:
                all_metrics.append(RAGASMetrics(
                    question=item.question,
                    difficulty=item.difficulty,
                    error=str(e),
                ))
```

- [ ] **Step 3: RAGASResult 新增 by_difficulty 属性**

在 `RAGASResult` 类中（第 145 行 `rich_table()` 之前）新增：

```python
    @property
    def by_difficulty(self) -> dict[str, "RAGASResult"]:
        """按难度分组返回各自的 RAGASResult。"""
        groups: dict[str, list[RAGASMetrics]] = {"简单": [], "中等": [], "复杂": [], "未分类": []}
        for m in self.metrics:
            key = m.difficulty or "未分类"
            groups.setdefault(key, []).append(m)
        result: dict[str, "RAGASResult"] = {}
        for diff, metrics_list in groups.items():
            if not metrics_list:
                continue
            result[diff] = RAGASResult(
                dataset_name=f"{self.dataset_name} ({diff})",
                metrics=metrics_list,
            )
        return result
```

- [ ] **Step 4: Commit**

```bash
git add src/eval/ragas_eval.py
git commit -m "feat(eval): RAGASMetrics 携带 difficulty + RAGASResult 按难度分组"
```

---

### Task 4: eval runner —— 串联分类 + 分组输出

**Files:**
- Modify: `src/eval/runner.py`

- [ ] **Step 1: run_eval 中调用 classify_difficulty**

在 `run_eval` 函数中，加载数据集后（第 133 行 `store = VectorStore()` 之前）插入：

```python
    # 自动分类难度
    console.print("[bold]正在自动分类问题难度...[/bold]")
    await dataset.classify_difficulty()
    dist = dataset.difficulty_distribution
    console.print(f"  难度分布: 简单={dist.get('简单', 0)}, 中等={dist.get('中等', 0)}, 复杂={dist.get('复杂', 0)}, 未分类={dist.get('未分类', 0)}")
    console.print()
```

- [ ] **Step 2: FullEvalReport.print() 追加难度分组表**

在 `print()` 方法末尾（第 55 行之后），在现有总览表后追加分组输出：

```python
            if self.ragas_result:
                console.print(self.ragas_result.rich_table())

        # 追加难度分组表
        self._print_by_difficulty(console)
```

新增 `_print_by_difficulty` 方法和更新 `print` 签名。完整修改：

在 `print()` 方法中，现有逻辑保持不变，末尾追加：

```python
            if self.ragas_result:
                console.print(self.ragas_result.rich_table())

        # ---- 难度分层 ----
        self._print_by_difficulty(console)
```

在 `FullEvalReport` 类中新增方法：

```python
    def _print_by_difficulty(self, console: Console) -> None:
        """按难度分组打印检索和 RAGAS 指标。"""
        retrieval_groups = self.retrieval_result.by_difficulty if self.retrieval_result else {}
        ragas_groups = self.ragas_result.by_difficulty if self.ragas_result else {}

        all_diffs = ["简单", "中等", "复杂", "未分类"]
        existing = [d for d in all_diffs if d in retrieval_groups or d in ragas_groups]
        if not existing:
            return

        # 检索分组表
        if retrieval_groups:
            rtable = Table(title="检索评测（按难度）")
            rtable.add_column("难度", style="cyan")
            rtable.add_column("题目数", justify="right")
            rtable.add_column("Recall@5", justify="right")
            rtable.add_column("MRR", justify="right")
            rtable.add_column("Hit@5", justify="right")

            for diff in all_diffs:
                g = retrieval_groups.get(diff)
                if g is None:
                    continue
                rtable.add_row(
                    diff,
                    str(len(g.metrics)),
                    f"{g.avg_recall.get(5, 0):.1%}",
                    f"{g.avg_mrr:.3f}",
                    f"{g.avg_hit.get(5, 0):.1%}",
                )
            console.print()
            console.print(rtable)

        # RAGAS 分组表
        if ragas_groups:
            gtable = Table(title="生成质量评测（按难度）")
            gtable.add_column("难度", style="cyan")
            gtable.add_column("题目数", justify="right")
            gtable.add_column("Faithfulness", justify="right")
            gtable.add_column("Answer Rel.", justify="right")
            gtable.add_column("Context Rel.", justify="right")

            for diff in all_diffs:
                g = ragas_groups.get(diff)
                if g is None:
                    continue
                gtable.add_row(
                    diff,
                    str(len(g.metrics)),
                    f"{g.avg_faithfulness:.1%}",
                    f"{g.avg_answer_relevance:.1%}",
                    f"{g.avg_context_relevance:.1%}",
                )
            console.print()
            console.print(gtable)
```

- [ ] **Step 3: FullEvalReport.to_dict() 追加分组数据**

在 `to_dict()` 方法末尾（第 100 行 `return report` 之前），`retrieval` 和 `ragas` 块之后插入：

```python
            # 难度分层
            retrieval_groups = self.retrieval_result.by_difficulty if self.retrieval_result else {}
            ragas_groups = self.ragas_result.by_difficulty if self.ragas_result else {}
            report["by_difficulty"] = {}
            for diff in ["简单", "中等", "复杂", "未分类"]:
                entry: dict = {"difficulty": diff}
                rg = retrieval_groups.get(diff)
                if rg:
                    entry["retrieval"] = {
                        "question_count": len(rg.metrics),
                        "recall_at_5": rg.avg_recall.get(5, 0),
                        "mrr": rg.avg_mrr,
                        "hit_at_5": rg.avg_hit.get(5, 0),
                    }
                gg = ragas_groups.get(diff)
                if gg:
                    entry["ragas"] = {
                        "question_count": len(gg.metrics),
                        "faithfulness": gg.avg_faithfulness,
                        "answer_relevance": gg.avg_answer_relevance,
                        "context_relevance": gg.avg_context_relevance,
                    }
                if rg or gg:
                    report["by_difficulty"][diff] = entry
```

- [ ] **Step 4: 验证 eval 流程正常**

```bash
cd C:\Users\87362\Desktop\agent\mc-rag && uv run python -m src.main eval -d eval_data/golden_50.json --ragas-only 2>&1 | head -80
```

- [ ] **Step 5: Commit**

```bash
git add src/eval/runner.py
git commit -m "feat(eval): run_eval 串联难度分类 + 分组表格输出"
```

---

### Task 5: Benchmark —— VersionResult 携带分层 RAGAS

**Files:**
- Modify: `src/benchmark/runner.py`

- [ ] **Step 1: VersionResult 新增 ragas_by_difficulty 字段**

在第 56-63 行 `VersionResult` dataclass 中加字段：

```python
@dataclass
class VersionResult:
    """单个版本的 benchmark 结果。"""
    agent_version: str
    timing: VersionTiming
    answers: list[str] = field(default_factory=list)
    ragas_faithfulness: float = 0.0
    ragas_answer_relevance: float = 0.0
    ragas_context_relevance: float = 0.0
    ragas_by_difficulty: dict[str, dict] = field(default_factory=dict)
    # {"简单": {"faithfulness": 0.85, ...}, "中等": {...}, "复杂": {...}}
```

- [ ] **Step 2: _run_version 中传入 difficulty 信息**

在 `_run_version` 方法中，需要将 `dataset.items` 的 difficulty 传入 `_compute_ragas_batch`。修改 `_compute_ragas_batch` 签名，增加 `difficulties` 参数：

```python
async def _compute_ragas_batch(dataset, answers, store, difficulties: list[str]) -> dict:
```

函数内部，在返回的 dict 中增加按难度分组的 RAGAS 指标：

```python
    # 在现有循环中收集 difficulty
    faith_by_diff: dict[str, list[float]] = {"简单": [], "中等": [], "复杂": [], "未分类": []}
    ar_by_diff: dict[str, list[float]] = {"简单": [], "中等": [], "复杂": [], "未分类": []}
    cr_by_diff: dict[str, list[float]] = {"简单": [], "中等": [], "复杂": [], "未分类": []}

    for idx, item in enumerate(dataset.items):
        question = item.question
        answer = answers[idx]
        diff = difficulties[idx] or "未分类"

        # ... 现有检索和计算逻辑 ...

        # 收集各指标到对应难度组
        faith_val = yes_count / len(statements) if statements else 1.0
        faith_by_diff[diff].append(faith_val)
        all_faith.append(faith_val)

        # Answer Relevance 计算后：
        ar_val = float(np.mean(np.dot(gen_norms, q_norm))) if reversed_qs else 0.0
        ar_by_diff[diff].append(ar_val)
        all_answer_rel.append(ar_val)

        # Context Relevance 计算后：
        cr_val = min(len(relevant_sents) / total_sents, 1.0) if total_sents > 0 and relevant_sents else 0.0
        cr_by_diff[diff].append(cr_val)
        all_context_rel.append(cr_val)

    ragas_by_difficulty = {}
    for diff in ["简单", "中等", "复杂", "未分类"]:
        if faith_by_diff.get(diff):
            ragas_by_difficulty[diff] = {
                "faithfulness": float(np.mean(faith_by_diff[diff])),
                "answer_relevance": float(np.mean(ar_by_diff[diff])),
                "context_relevance": float(np.mean(cr_by_diff[diff])),
            }

    return {
        "avg_faithfulness": ...,
        "avg_answer_relevance": ...,
        "avg_context_relevance": ...,
        "by_difficulty": ragas_by_difficulty,
    }
```

调用处（`_run_version` 方法中）传入 difficulties：

```python
            difficulties = [item.difficulty for item in dataset.items]
            ragas_metrics = await _compute_ragas_batch(dataset, answers, store, difficulties)
            faithfulness = ragas_metrics["avg_faithfulness"]
            answer_relevance = ragas_metrics["avg_answer_relevance"]
            context_relevance = ragas_metrics["avg_context_relevance"]

        return VersionResult(
            agent_version=version,
            timing=vtiming,
            answers=answers,
            ragas_faithfulness=faithfulness,
            ragas_answer_relevance=answer_relevance,
            ragas_context_relevance=context_relevance,
            ragas_by_difficulty=ragas_metrics.get("by_difficulty", {}),
        )
```

- [ ] **Step 3: _persist_runs 写入分层指标**

在 `_persist_runs` 中（第 226-248 行），`run_data` dict 增加分层字段：

```python
            run_data = {
                # ... 现有字段 ...
                "faithfulness": vr.ragas_faithfulness,
                "answer_relevance": vr.ragas_answer_relevance,
                "context_relevance": vr.ragas_context_relevance,
            }
            # 追加分层指标
            for diff in ["简单", "中等", "复杂"]:
                bd = vr.ragas_by_difficulty.get(diff, {})
                run_data[f"faithfulness_{diff}"] = bd.get("faithfulness", 0.0)
                run_data[f"answer_relevance_{diff}"] = bd.get("answer_relevance", 0.0)
                run_data[f"context_relevance_{diff}"] = bd.get("context_relevance", 0.0)
```

- [ ] **Step 4: _build_report_dict 含分层数据**

在 `_build_report_dict`（第 381-409 行）的 `"ragas"` 块中增加 `by_difficulty`：

```python
        "ragas": {
            "faithfulness": vr.ragas_faithfulness,
            "answer_relevance": vr.ragas_answer_relevance,
            "context_relevance": vr.ragas_context_relevance,
            "by_difficulty": vr.ragas_by_difficulty,
        },
```

- [ ] **Step 5: BenchmarkRunner.run() 中调用 classify_difficulty**

在 `run()` 方法中，`store = VectorStore()` 之后、检索评测之前：

```python
        dataset = EvalDataset.from_json(self.dataset_path)
        store = VectorStore()

        # 自动分类难度
        await dataset.classify_difficulty()
```

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/runner.py
git commit -m "feat(benchmark): VersionResult 携带分层 RAGAS + classify_difficulty 串联"
```

---

### Task 6: Benchmark Reporter —— 按难度展开对比表

**Files:**
- Modify: `src/benchmark/reporter.py`

- [ ] **Step 1: print_report 追加难度分层对比表**

在 `print_report` 函数（第 16-73 行），现有总表之后、趋势摘要之前插入难度分层表。在最后的 `console.print()` 之前：

```python
    # ---- 难度分层对比表 ----
    diffs = ["简单", "中等", "复杂"]
    # 收集每个版本的每个难度下有哪些题目数
    difficulty_counts: dict[str, dict[str, int]] = {}  # {version: {diff: count}}
    for vr in report.versions:
        difficulty_counts[vr.agent_version] = {}
        for diff in diffs:
            bd = vr.ragas_by_difficulty.get(diff, {})
            difficulty_counts[vr.agent_version][diff] = bd.get("question_count", 0)

    for diff in diffs:
        counts = [difficulty_counts.get(vr.agent_version, {}).get(diff, 0) for vr in report.versions]
        if all(c == 0 for c in counts):
            continue
        dt = Table(title=f"难度: {diff}")
        dt.add_column("指标", style="cyan")
        for vr in report.versions:
            dt.add_column(vr.agent_version.upper(), justify="right")

        _add_metric_row(dt, "Faithfulness",
            [vr.ragas_by_difficulty.get(diff, {}).get("faithfulness", 0) for vr in report.versions],
            fmt=".1%")
        _add_metric_row(dt, "Answer Relevance",
            [vr.ragas_by_difficulty.get(diff, {}).get("answer_relevance", 0) for vr in report.versions],
            fmt=".1%")
        _add_metric_row(dt, "Context Relevance",
            [vr.ragas_by_difficulty.get(diff, {}).get("context_relevance", 0) for vr in report.versions],
            fmt=".1%")

        console.print()
        console.print(dt)
```

- [ ] **Step 2: save_json_report 含分层数据**

在 `save_json_report`（第 103-142 行）的每个 `vr` 循环中，`ragas` 块增加 `by_difficulty`：

```python
            "ragas": {
                "faithfulness": vr.ragas_faithfulness,
                "answer_relevance": vr.ragas_answer_relevance,
                "context_relevance": vr.ragas_context_relevance,
                "by_difficulty": vr.ragas_by_difficulty,
            },
```

- [ ] **Step 3: Commit**

```bash
git add src/benchmark/reporter.py
git commit -m "feat(benchmark): Reporter 按难度展开版本对比表"
```

---

### Task 7: Benchmark Comparator —— 基线检测含分层指标

**Files:**
- Modify: `src/benchmark/comparator.py`

- [ ] **Step 1: _check_version 中增加分层指标检查**

在 `_check_version` 函数（第 27-47 行）末尾，现有检查之后增加：

```python
    # 难度分层指标检查
    for diff in ["简单", "中等", "复杂"]:
        bd = vr.ragas_by_difficulty.get(diff, {})
        if not bd:
            continue
        for metric in ["faithfulness", "answer_relevance", "context_relevance"]:
            value = bd.get(metric, 0)
            cfg = baselines.get(f"{metric}_{diff}")
            if cfg is None:
                cfg = baselines.get(metric)  # fallback 到通用阈值
            if cfg is not None and value < cfg["min_threshold"]:
                all_ok = False
```

- [ ] **Step 2: Commit**

```bash
git add src/benchmark/comparator.py
git commit -m "feat(benchmark): Comparator 基线检测含分层指标"
```

---

### Task 8: Benchmark Database —— 表结构加难度分层列

**Files:**
- Modify: `src/benchmark/database.py`

- [ ] **Step 1: SCHEMA 新增 9 列**

在 `SCHEMA` 字符串（第 10-40 行）的 `context_relevance` 行之后、`passed` 行之前插入：

```sql
    faithfulness_简单 REAL DEFAULT 0.0,
    faithfulness_中等 REAL DEFAULT 0.0,
    faithfulness_复杂 REAL DEFAULT 0.0,
    answer_relevance_简单 REAL DEFAULT 0.0,
    answer_relevance_中等 REAL DEFAULT 0.0,
    answer_relevance_复杂 REAL DEFAULT 0.0,
    context_relevance_简单 REAL DEFAULT 0.0,
    context_relevance_中等 REAL DEFAULT 0.0,
    context_relevance_复杂 REAL DEFAULT 0.0,
```

- [ ] **Step 2: insert_run 更新 columns 列表**

在 `insert_run` 方法（第 69-91 行）的 `columns` 列表中，`context_relevance` 之后插入：

```python
        columns = [
            "id", "timestamp", "agent_version", "dataset_name",
            "question_count", "git_commit", "total_duration_ms",
            "avg_ttft_ms", "avg_retrieval_ms", "avg_generation_ms",
            "recall_at_5", "precision_at_5", "mrr", "ndcg_at_5", "hit_at_5",
            "faithfulness", "answer_relevance", "context_relevance",
            "faithfulness_简单", "faithfulness_中等", "faithfulness_复杂",
            "answer_relevance_简单", "answer_relevance_中等", "answer_relevance_复杂",
            "context_relevance_简单", "context_relevance_中等", "context_relevance_复杂",
            "passed", "report_json",
        ]
```

- [ ] **Step 3: 验证 DB 迁移后插入正常**

由于 SQLite `CREATE TABLE IF NOT EXISTS` 不会自动加列，对已有数据库需要手动处理。在 `_init_schema` 中增加迁移逻辑：

```python
    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
                # 迁移：为已有表补充分层列
                existing_cols = {row[1] for row in conn.execute(
                    "PRAGMA table_info(benchmark_runs)"
                ).fetchall()}
                diff_cols = [
                    "faithfulness_简单", "faithfulness_中等", "faithfulness_复杂",
                    "answer_relevance_简单", "answer_relevance_中等", "answer_relevance_复杂",
                    "context_relevance_简单", "context_relevance_中等", "context_relevance_复杂",
                ]
                for col in diff_cols:
                    if col not in existing_cols:
                        conn.execute(
                            f"ALTER TABLE benchmark_runs ADD COLUMN {col} REAL DEFAULT 0.0"
                        )
                conn.commit()
        except Exception as e:
            raise RuntimeError(
                f"无法初始化 benchmark 数据库 schema（路径: {self.db_path}）：{e}"
            ) from e
```

- [ ] **Step 4: get_history 查询增加分层列**

按当前设计，history 表查询保持现有列即可（过于详细的列会让表格过宽）。不修改 `get_history`。

- [ ] **Step 5: 验证数据库迁移**

```bash
cd C:\Users\87362\Desktop\agent\mc-rag && uv run python -c "
from src.benchmark.database import BenchmarkDB
db = BenchmarkDB()
db.init_default_baselines()
print('DB 初始化成功')
cols = db._connect().execute('PRAGMA table_info(benchmark_runs)').fetchall()
for c in cols:
    print(f'  {c[1]}')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/database.py
git commit -m "feat(benchmark): DB 表加难度分层列 + 自动迁移"
```

---

### Task 9: 端到端验证

**Files:** 无

- [ ] **Step 1: 验证 eval --ragas-only 流程**

```bash
cd C:\Users\87362\Desktop\agent\mc-rag && uv run python -m src.main eval -d eval_data/golden_50.json --ragas-only
```

预期：难度分布打印 → 逐题评测 → 总表 → 按难度分组表

- [ ] **Step 2: 验证 benchmark 流程**

```bash
cd C:\Users\87362\Desktop\agent\mc-rag && uv run python -m src.main benchmark --versions v1,v4
```

预期：难度分布打印 → 检索评测 → v1 RAGAS → v4 RAGAS → 版本对比总表 → 难度分层对比表

- [ ] **Step 3: 验证 JSON 报告含分层数据**

```bash
cd C:\Users\87362\Desktop\agent\mc-rag && uv run python -m src.main eval -d eval_data/golden_50.json --ragas-only -o test_report.json && uv run python -c "
import json
with open('test_report.json', encoding='utf-8') as f:
    d = json.load(f)
print('by_difficulty keys:', list(d.get('by_difficulty', {}).keys()))
"
```

- [ ] **Step 4: 清理测试产物**

```bash
rm -f C:\Users\87362\Desktop\agent\mc-rag\test_report.json
```

- [ ] **Step 5: Commit（如有修改）**

如果有修复性修改，提交：

```bash
git add -A
git commit -m "fix: 端到端验证修复"
```
