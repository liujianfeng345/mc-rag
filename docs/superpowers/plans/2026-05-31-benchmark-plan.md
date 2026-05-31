# Benchmark 测试系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 `src/benchmark/` 模块，实现跨版本对比、性能打点、回归检测的完整 benchmark 体系

**Architecture:** 8 个文件，逐层构建——config → database → profiler → runner → comparator → reporter → cli → main 集成。benchmark 调用现有 `src/eval/` 的 `RetrievalEvaluator` 做检索评测，RAGAS 生成质量评测逻辑在 runner 内联实现（因为需要用各版本 agent 图的实际输出而非通用 LLM 生成答案）。性能打点通过 `graph.astream_events` 捕获首 token 时间。

**Tech Stack:** Python 3.11+, sqlite3, Rich, langgraph astream_events, 现有 eval 模块

---

### Task 1: 创建 package 和配置模块

**Files:**
- Create: `src/benchmark/__init__.py`
- Create: `src/benchmark/benchmark_config.py`

- [ ] **Step 1: 创建 `__init__.py`**

```python
# src/benchmark/__init__.py
"""Benchmark 测试系统 — 跨版本对比、性能打点、回归检测。"""
```

- [ ] **Step 2: 创建 `benchmark_config.py`**

```python
# src/benchmark/benchmark_config.py
"""Benchmark 默认配置。"""

from pathlib import Path

# 默认评测版本
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4"]

# 默认数据集路径
DEFAULT_DATASET = str(Path(__file__).parent.parent.parent / "eval_data" / "generated_questions.json")

# 检索指标 K 值
RETRIEVAL_K_VALUES = [3, 5, 10]

# 是否启用阶段拆解打点
PROFILE_STAGES = True

# SQLite 数据库路径
DB_PATH = str(Path(__file__).parent.parent.parent / "benchmark_results" / "benchmark.db")

# JSON 报告输出目录
REPORT_DIR = str(Path(__file__).parent.parent.parent / "benchmark_results")

# 默认阈值（首次运行时写入 benchmark_baselines 表）
DEFAULT_THRESHOLDS = {
    "recall_at_5":       {"min": 0.6, "max_degradation_pct": 15},
    "mrr":               {"min": 0.5, "max_degradation_pct": 15},
    "hit_at_5":          {"min": 0.7, "max_degradation_pct": 10},
    "faithfulness":      {"min": 0.6, "max_degradation_pct": 15},
    "answer_relevance":  {"min": 0.6, "max_degradation_pct": 15},
    "context_relevance": {"min": 0.6, "max_degradation_pct": 15},
}
```

- [ ] **Step 3: 将 `benchmark_results/` 加入 `.gitignore`**

```bash
echo "benchmark_results/" >> .gitignore
```

- [ ] **Step 4: 验证 Python 语法**

```bash
uv run python -c "from src.benchmark.benchmark_config import DEFAULT_VERSIONS; print(DEFAULT_VERSIONS)"
```
预期: `['v1', 'v2', 'v3', 'v4']`

- [ ] **Step 5: 提交**

```bash
git add src/benchmark/__init__.py src/benchmark/benchmark_config.py
git commit -m "feat(benchmark): 新增 benchmark 模块包和配置

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: 数据库模块

**Files:**
- Create: `src/benchmark/database.py`

- [ ] **Step 1: 创建 `database.py`**

```python
# src/benchmark/database.py
"""SQLite 数据库：建表、CRUD、历史查询。"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_config import DB_PATH, DEFAULT_THRESHOLDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    git_commit TEXT DEFAULT '',
    total_duration_ms REAL DEFAULT 0.0,
    avg_ttft_ms REAL DEFAULT 0.0,
    avg_retrieval_ms REAL DEFAULT 0.0,
    avg_generation_ms REAL DEFAULT 0.0,
    recall_at_5 REAL DEFAULT 0.0,
    precision_at_5 REAL DEFAULT 0.0,
    mrr REAL DEFAULT 0.0,
    ndcg_at_5 REAL DEFAULT 0.0,
    hit_at_5 REAL DEFAULT 0.0,
    faithfulness REAL DEFAULT 0.0,
    answer_relevance REAL DEFAULT 0.0,
    context_relevance REAL DEFAULT 0.0,
    passed INTEGER DEFAULT 0,
    report_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS benchmark_baselines (
    metric_name TEXT PRIMARY KEY,
    min_threshold REAL NOT NULL,
    max_degradation_pct REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class BenchmarkDB:
    """Benchmark 数据存储。"""

    def __init__(self, db_path: str = ""):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # ---- benchmark_runs CRUD ----

    def insert_run(self, run_data: dict) -> str:
        """插入一次 benchmark 记录，返回 run_id。"""
        columns = [
            "id", "timestamp", "agent_version", "dataset_name",
            "question_count", "git_commit", "total_duration_ms",
            "avg_ttft_ms", "avg_retrieval_ms", "avg_generation_ms",
            "recall_at_5", "precision_at_5", "mrr", "ndcg_at_5", "hit_at_5",
            "faithfulness", "answer_relevance", "context_relevance",
            "passed", "report_json",
        ]
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT OR REPLACE INTO benchmark_runs ({', '.join(columns)}) VALUES ({placeholders})"

        with self._connect() as conn:
            conn.execute(sql, tuple(run_data.get(c, 0 if c != "report_json" else "{}") for c in columns))
            conn.commit()
        return run_data["id"]

    def get_last_run(self, agent_version: str, dataset_name: str) -> dict | None:
        """获取指定版本+数据集的上一次跑分。"""
        sql = """
            SELECT * FROM benchmark_runs
            WHERE agent_version = ? AND dataset_name = ?
            ORDER BY timestamp DESC LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (agent_version, dataset_name)).fetchone()
            return dict(row) if row else None

    def get_history(self, agent_version: str = "", limit: int = 10) -> list[dict]:
        """获取历史跑分，可按版本过滤。"""
        if agent_version:
            sql = """
                SELECT id, timestamp, agent_version, dataset_name, recall_at_5, mrr,
                       hit_at_5, faithfulness, answer_relevance, avg_ttft_ms, passed
                FROM benchmark_runs
                WHERE agent_version = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            params = (agent_version, limit)
        else:
            sql = """
                SELECT id, timestamp, agent_version, dataset_name, recall_at_5, mrr,
                       hit_at_5, faithfulness, answer_relevance, avg_ttft_ms, passed
                FROM benchmark_runs
                ORDER BY timestamp DESC LIMIT ?
            """
            params = (limit,)
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    # ---- benchmark_baselines CRUD ----

    def get_baselines(self) -> dict[str, dict]:
        """返回所有基线阈值，key 为 metric_name。"""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM benchmark_baselines").fetchall()
        return {row["metric_name"]: {"min_threshold": row["min_threshold"],
                                      "max_degradation_pct": row["max_degradation_pct"]}
                for row in rows}

    def set_baseline(self, metric_name: str, min_threshold: float, max_degradation_pct: float) -> None:
        """写入或更新一条基线阈值。"""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT OR REPLACE INTO benchmark_baselines (metric_name, min_threshold, max_degradation_pct, updated_at)
            VALUES (?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (metric_name, min_threshold, max_degradation_pct, now))
            conn.commit()

    def init_default_baselines(self) -> None:
        """首次运行时用默认阈值填充基线表（不覆盖已有数据）。"""
        existing = self.get_baselines()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for name, cfg in DEFAULT_THRESHOLDS.items():
                if name not in existing:
                    conn.execute(
                        "INSERT INTO benchmark_baselines (metric_name, min_threshold, max_degradation_pct, updated_at) VALUES (?, ?, ?, ?)",
                        (name, cfg["min"], cfg["max_degradation_pct"], now),
                    )
            conn.commit()
```

- [ ] **Step 2: 验证数据库初始化**

```bash
uv run python -c "
from src.benchmark.database import BenchmarkDB
db = BenchmarkDB()
db.init_default_baselines()
print(db.get_baselines())
print(db.get_history())
"
```
预期: 打印 6 条基线阈值，空历史列表

- [ ] **Step 3: 提交**

```bash
git add src/benchmark/database.py
git commit -m "feat(benchmark): 新增 SQLite 数据库模块

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: 性能打点模块

**Files:**
- Create: `src/benchmark/profiler.py`

- [ ] **Step 1: 创建 `profiler.py`**

```python
# src/benchmark/profiler.py
"""性能打点 — 端到端耗时、阶段拆解、首 token 延迟。"""

import time
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class QuestionTiming:
    """单条问题的耗时明细（毫秒）。"""
    question: str = ""
    retrieval_ms: float = 0.0
    ttft_ms: float = 0.0       # 首 token 到达时间
    generation_ms: float = 0.0  # 首 token → 末 token
    total_ms: float = 0.0


@dataclass
class VersionTiming:
    """单个版本的耗时汇总。"""
    agent_version: str = ""
    per_question: list[QuestionTiming] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        vals = [t.ttft_ms for t in self.per_question if t.ttft_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_retrieval_ms(self) -> float:
        vals = [t.retrieval_ms for t in self.per_question if t.retrieval_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_generation_ms(self) -> float:
        vals = [t.generation_ms for t in self.per_question if t.generation_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_total_ms(self) -> float:
        vals = [t.total_ms for t in self.per_question]
        return sum(vals) / len(vals) if vals else 0.0


async def run_with_timing(
    graph,
    input_state: dict,
    output_node: str,
) -> tuple[dict, QuestionTiming]:
    """执行一次 agent graph 调用并打点。

    通过 astream_events 捕获流式事件：
    - 第一个 on_chat_model_stream 事件 → 记录首 token 时间
    - 最后一个 on_chat_model_stream 事件 → 记录生成结束时间
    - on_chain_end → 收集最终 state

    返回：
        (final_state, timing_info)
    """
    timing = QuestionTiming()
    t_start = time.perf_counter()

    final_state: dict = {}
    first_token_seen = False
    t_first_token = t_start
    t_last_token = t_start

    async for event in graph.astream_events(input_state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node == output_node:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    if not first_token_seen:
                        t_first_token = time.perf_counter()
                        first_token_seen = True
                    t_last_token = time.perf_counter()

        if kind == "on_chain_end" and isinstance(
            event.get("data", {}).get("output"), dict
        ):
            final_state.update(event["data"]["output"])

    t_end = time.perf_counter()

    timing.total_ms = (t_end - t_start) * 1000
    if first_token_seen:
        timing.ttft_ms = (t_first_token - t_start) * 1000
        timing.generation_ms = (t_last_token - t_first_token) * 1000

    return final_state, timing


async def measure_retrieval(
    store,
    question: str,
    top_k: int = 5,
) -> tuple[list, float]:
    """单独测量一次检索耗时，返回 (文档列表, 毫秒)。"""
    t0 = time.perf_counter()
    docs = await store.hybrid_search(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return docs, elapsed_ms
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -c "from src.benchmark.profiler import QuestionTiming, VersionTiming, run_with_timing; print('ok')"
```
预期: `ok`

- [ ] **Step 3: 提交**

```bash
git add src/benchmark/profiler.py
git commit -m "feat(benchmark): 新增性能打点模块

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Runner 核心调度模块

**Files:**
- Create: `src/benchmark/runner.py`

- [ ] **Step 1: 创建 `runner.py`**

```python
# src/benchmark/runner.py
"""Benchmark 核心调度 — 多版本循环、调用 agent 图、汇总结果。"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..eval.dataset import EvalDataset
from ..eval.retrieval import RetrievalEvaluator, EvalResult as RetrievalResult
from ..vector.vector_store import VectorStore
from ..utils.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, RETRIEVAL_TOP_K,
)

from .benchmark_config import DEFAULT_VERSIONS, RETRIEVAL_K_VALUES
from .database import BenchmarkDB
from .profiler import (
    QuestionTiming, VersionTiming, run_with_timing, measure_retrieval,
)

# 各版本最终答案节点名
OUTPUT_NODES = {"v4": "synthesize", "v3": "synthesize", "v2": "generate", "v1": "generate"}


def _build_graph_for_version(version: str, store: VectorStore):
    """按版本字符串动态导入并构建 agent 图。"""
    if version == "v4":
        from ..agent_v4.graph import build_rag_graph
    elif version == "v3":
        from ..agent_v3.graph import build_rag_graph
    elif version == "v2":
        from ..agent_v2.graph import build_rag_graph
    else:
        from ..agent.graph import build_rag_graph
    return build_rag_graph(store)


def _get_git_commit() -> str:
    """获取当前 HEAD commit hash。"""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


@dataclass
class VersionResult:
    """单个版本的 benchmark 结果。"""
    agent_version: str
    timing: VersionTiming
    answers: list[str] = field(default_factory=list)
    ragas_faithfulness: float = 0.0
    ragas_answer_relevance: float = 0.0
    ragas_context_relevance: float = 0.0


@dataclass
class BenchmarkReport:
    """完整 benchmark 报告。"""
    run_id: str
    dataset_name: str
    question_count: int
    git_commit: str
    retrieval: RetrievalResult | None = None
    versions: list[VersionResult] = field(default_factory=list)
    passed_map: dict[str, bool] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0


class BenchmarkRunner:
    """Benchmark 调度器。"""

    def __init__(
        self,
        dataset_path: str,
        versions: list[str] | None = None,
        profile_only: bool = False,
        eval_only: bool = False,
    ):
        self.dataset_path = dataset_path
        self.versions = versions or DEFAULT_VERSIONS
        self.profile_only = profile_only
        self.eval_only = eval_only
        self.db = BenchmarkDB()

    async def run(self) -> BenchmarkReport:
        """执行完整 benchmark 流程。"""
        t_total_start = time.perf_counter()

        dataset = EvalDataset.from_json(self.dataset_path)
        store = VectorStore()
        db = self.db
        db.init_default_baselines()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        git_commit = _get_git_commit()

        report = BenchmarkReport(
            run_id=run_id,
            dataset_name=dataset.name,
            question_count=len(dataset),
            git_commit=git_commit,
        )

        # ---- 检索评测（版本无关，跑一次） ----
        if not self.profile_only:
            retrieval_eval = RetrievalEvaluator(store)
            report.retrieval = await retrieval_eval.evaluate(
                dataset, k_values=RETRIEVAL_K_VALUES, verbose=False,
            )

        # ---- 逐版本跑 ----
        for version in self.versions:
            vresult = await self._run_version(version, dataset, store)
            report.versions.append(vresult)

        # ---- 基线对比 ----
        from .comparator import compare_all
        report.passed_map = compare_all(db, report.retrieval, report.versions, report.dataset_name)

        # ---- 写入数据库 ----
        self._persist_runs(db, report)

        report.total_elapsed_ms = (time.perf_counter() - t_total_start) * 1000
        return report

    async def _run_version(
        self, version: str, dataset, store: VectorStore,
    ) -> VersionResult:
        """对单个版本跑全部问题。"""
        graph = _build_graph_for_version(version, store)
        output_node = OUTPUT_NODES.get(version, "generate")

        timings: list[QuestionTiming] = []
        answers: list[str] = []

        for item in dataset.items:
            # 构建输入 state
            input_state: dict = {"question": item.question}
            if version in ("v1", "v2"):
                input_state["rewrite_count"] = 0

            if self.eval_only:
                # eval_only 模式：直接用 graph.ainvoke（不打点）
                result = await graph.ainvoke(input_state)
                answer = result.get("final_report") or result.get("generation", "")
                timing = QuestionTiming(question=item.question)
            else:
                # 正常模式：打点运行
                final_state, timing = await run_with_timing(graph, input_state, output_node)
                answer = final_state.get("final_report") or final_state.get("generation", "")
                timing.question = item.question

            timings.append(timing)
            answers.append(answer)

        vtiming = VersionTiming(agent_version=version, per_question=timings)

        # RAGAS 生成质量评测
        faithfulness = 0.0
        answer_relevance = 0.0
        context_relevance = 0.0
        if not self.profile_only:
            ragas_metrics = await _compute_ragas_batch(dataset, answers, store)
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
        )

    def _persist_runs(self, db: BenchmarkDB, report: BenchmarkReport) -> None:
        """将报告写入 SQLite。"""
        retrieval = report.retrieval
        for vr in report.versions:
            run_data = {
                "id": f"{report.run_id}-{vr.agent_version}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_version": vr.agent_version,
                "dataset_name": report.dataset_name,
                "question_count": report.question_count,
                "git_commit": report.git_commit,
                "total_duration_ms": vr.timing.avg_total_ms,
                "avg_ttft_ms": vr.timing.avg_ttft_ms,
                "avg_retrieval_ms": vr.timing.avg_retrieval_ms,
                "avg_generation_ms": vr.timing.avg_generation_ms,
                "recall_at_5": retrieval.avg_recall.get(5, 0) if retrieval else 0,
                "precision_at_5": retrieval.avg_precision.get(5, 0) if retrieval else 0,
                "mrr": retrieval.avg_mrr if retrieval else 0,
                "ndcg_at_5": retrieval.avg_ndcg.get(5, 0) if retrieval else 0,
                "hit_at_5": retrieval.avg_hit.get(5, 0) if retrieval else 0,
                "faithfulness": vr.ragas_faithfulness,
                "answer_relevance": vr.ragas_answer_relevance,
                "context_relevance": vr.ragas_context_relevance,
                "passed": 1 if report.passed_map.get(vr.agent_version, False) else 0,
                "report_json": json.dumps(_build_report_dict(report, vr), ensure_ascii=False),
            }
            db.insert_run(run_data)


# ---- RAGAS 评测逻辑（内联，复用 ragas_eval.py 的 prompt 和计算方式） ----

from ..eval.ragas_eval import (
    FAITHFULNESS_STATEMENTS_PROMPT,
    FAITHFULNESS_VERIFY_PROMPT,
    ANSWER_RELEVANCE_REVERSE_PROMPT,
    CONTEXT_RELEVANCE_EXTRACT_PROMPT,
)

import numpy as np
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


async def _compute_ragas_batch(dataset, answers, store) -> dict:
    """批量计算 RAGAS 指标（Faithfulness, Answer Relevance, Context Relevance）。"""
    llm = ChatOpenAI(
        model=LLM_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        temperature=0.0, max_tokens=2048,
    )
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    all_faith = []
    all_answer_rel = []
    all_context_rel = []

    for idx, item in enumerate(dataset.items):
        question = item.question
        answer = answers[idx]

        docs = await store.hybrid_search(question, top_k=RETRIEVAL_TOP_K)
        context = "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}" for d in docs
        )

        # Faithfulness
        statements = await _extract_statements(llm, answer)
        if statements:
            verdicts = await asyncio.gather(*[
                _verify_statement(llm, s, context) for s in statements
            ])
            yes_count = sum(1 for v in verdicts if v == "YES")
            all_faith.append(yes_count / len(statements))
        else:
            all_faith.append(1.0)

        # Answer Relevance
        reversed_qs = await _reverse_questions(llm, answer)
        if reversed_qs:
            all_texts = [question] + reversed_qs
            all_embeds = await asyncio.to_thread(embeddings.embed_documents, all_texts)
            q_embed = np.array(all_embeds[0])
            gen_embeds = np.array(all_embeds[1:])
            q_norm = q_embed / (np.linalg.norm(q_embed) + 1e-8)
            gen_norms = gen_embeds / (np.linalg.norm(gen_embeds, axis=1, keepdims=True) + 1e-8)
            all_answer_rel.append(float(np.mean(np.dot(gen_norms, q_norm))))
        else:
            all_answer_rel.append(0.0)

        # Context Relevance
        relevant_sents = await _extract_relevant(llm, question, context[:8000])
        total_sents = len([s for s in re.split(r'[。！？\n]+', context) if s.strip()])
        if total_sents > 0 and relevant_sents:
            all_context_rel.append(min(len(relevant_sents) / total_sents, 1.0))
        else:
            all_context_rel.append(0.0)

    return {
        "avg_faithfulness": float(np.mean(all_faith)) if all_faith else 0.0,
        "avg_answer_relevance": float(np.mean(all_answer_rel)) if all_answer_rel else 0.0,
        "avg_context_relevance": float(np.mean(all_context_rel)) if all_context_rel else 0.0,
    }


def _parse_json(text: str) -> dict | list:
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _parse_json_array(text: str) -> list:
    result = _parse_json(text)
    return result if isinstance(result, list) else []


async def _extract_statements(llm, answer: str) -> list[str]:
    try:
        msg = await llm.ainvoke([HumanMessage(content=FAITHFULNESS_STATEMENTS_PROMPT.format(answer=answer))])
        return _parse_json_array(msg.content)
    except Exception:
        return []


async def _verify_statement(llm, statement: str, context: str) -> str:
    try:
        msg = await llm.ainvoke([HumanMessage(content=FAITHFULNESS_VERIFY_PROMPT.format(statement=statement, context=context))])
        result = _parse_json(msg.content)
        return result.get("verdict", "NO") if isinstance(result, dict) else "NO"
    except Exception:
        return "NO"


async def _reverse_questions(llm, answer: str) -> list[str]:
    try:
        msg = await llm.ainvoke([HumanMessage(content=ANSWER_RELEVANCE_REVERSE_PROMPT.format(answer=answer))])
        return _parse_json_array(msg.content)
    except Exception:
        return []


async def _extract_relevant(llm, question: str, context: str) -> list[str]:
    try:
        msg = await llm.ainvoke([HumanMessage(content=CONTEXT_RELEVANCE_EXTRACT_PROMPT.format(question=question, context=context))])
        return _parse_json_array(msg.content)
    except Exception:
        return []


def _build_report_dict(report: BenchmarkReport, vr: VersionResult) -> dict:
    """构建单个版本的报告字典（用于 JSON 快照）。"""
    r = report.retrieval
    return {
        "run_id": report.run_id,
        "agent_version": vr.agent_version,
        "dataset_name": report.dataset_name,
        "question_count": report.question_count,
        "git_commit": report.git_commit,
        "timing": {
            "avg_total_ms": vr.timing.avg_total_ms,
            "avg_ttft_ms": vr.timing.avg_ttft_ms,
            "avg_retrieval_ms": vr.timing.avg_retrieval_ms,
            "avg_generation_ms": vr.timing.avg_generation_ms,
        },
        "retrieval": {
            "recall_at_5": r.avg_recall.get(5, 0) if r else 0,
            "precision_at_5": r.avg_precision.get(5, 0) if r else 0,
            "mrr": r.avg_mrr if r else 0,
            "ndcg_at_5": r.avg_ndcg.get(5, 0) if r else 0,
            "hit_at_5": r.avg_hit.get(5, 0) if r else 0,
        },
        "ragas": {
            "faithfulness": vr.ragas_faithfulness,
            "answer_relevance": vr.ragas_answer_relevance,
            "context_relevance": vr.ragas_context_relevance,
        },
        "passed": report.passed_map.get(vr.agent_version, False),
    }
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -c "from src.benchmark.runner import BenchmarkRunner, BenchmarkReport, VersionResult; print('ok')"
```
预期: `ok`

- [ ] **Step 3: 提交**

```bash
git add src/benchmark/runner.py
git commit -m "feat(benchmark): 新增核心调度 Runner 模块

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: 基线对比模块

**Files:**
- Create: `src/benchmark/comparator.py`

- [ ] **Step 1: 创建 `comparator.py`**

```python
# src/benchmark/comparator.py
"""基线对比 — 阈值检查、退化检测。"""

from ..eval.retrieval import EvalResult as RetrievalResult
from .database import BenchmarkDB
from .runner import VersionResult


def compare_all(
    db: BenchmarkDB,
    retrieval: RetrievalResult | None,
    versions: list[VersionResult],
    dataset_name: str = "",
) -> dict[str, bool]:
    """对所有版本执行基线对比，返回 {version: passed}。"""
    baselines = db.get_baselines()
    if not baselines:
        return {vr.agent_version: True for vr in versions}

    passed_map: dict[str, bool] = {}
    for vr in versions:
        last_run = db.get_last_run(vr.agent_version, dataset_name)
        passed_map[vr.agent_version] = _check_version(vr, retrieval, baselines, last_run)
    return passed_map


def _check_version(
    vr: VersionResult,
    retrieval: RetrievalResult | None,
    baselines: dict,
    last_run: dict | None,
) -> bool:
    """检查单个版本是否通过——同时验证绝对阈值和相对退化。"""
    all_ok = True

    # 检索指标
    if retrieval:
        all_ok &= _check_metric("recall_at_5", retrieval.avg_recall.get(5, 0), baselines, last_run)
        all_ok &= _check_metric("mrr", retrieval.avg_mrr, baselines, last_run)
        all_ok &= _check_metric("hit_at_5", retrieval.avg_hit.get(5, 0), baselines, last_run)

    # RAGAS 指标
    all_ok &= _check_metric("faithfulness", vr.ragas_faithfulness, baselines, last_run)
    all_ok &= _check_metric("answer_relevance", vr.ragas_answer_relevance, baselines, last_run)
    all_ok &= _check_metric("context_relevance", vr.ragas_context_relevance, baselines, last_run)

    return all_ok


def _check_metric(name: str, value: float, baselines: dict, last_run: dict | None) -> bool:
    """检查单个指标：绝对阈值 + 相对退化。"""
    cfg = baselines.get(name)
    if cfg is None:
        return True
    min_threshold = cfg["min_threshold"]
    if value < min_threshold:
        return False

    # 检查相比上次的退化
    if last_run is not None:
        prev_val = last_run.get(name, None)
        if prev_val is not None and prev_val > 0:
            degrade_pct = (prev_val - value) / prev_val * 100
            if degrade_pct > cfg["max_degradation_pct"]:
                return False

    return True


def get_trend(
    db: BenchmarkDB,
    agent_version: str,
    dataset_name: str,
    current: dict,
) -> dict:
    """对比本次跑分与上次，返回趋势信息。"""
    last = db.get_last_run(agent_version, dataset_name)
    if last is None:
        return {"is_new": True}

    metrics = ["recall_at_5", "mrr", "hit_at_5", "faithfulness", "answer_relevance", "avg_ttft_ms"]
    trends = {}
    for m in metrics:
        cur_val = current.get(m, 0)
        prev_val = last.get(m, 0)
        if prev_val == 0:
            trends[m] = {"prev": prev_val, "cur": cur_val, "delta": 0, "pct": 0}
        else:
            delta = cur_val - prev_val
            trends[m] = {"prev": prev_val, "cur": cur_val, "delta": delta, "pct": delta / prev_val * 100}
    return {"is_new": False, "trends": trends}
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -c "from src.benchmark.comparator import compare_all, get_trend; print('ok')"
```
预期: `ok`

- [ ] **Step 3: 提交**

```bash
git add src/benchmark/comparator.py
git commit -m "feat(benchmark): 新增基线对比模块

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: 报告输出模块

**Files:**
- Create: `src/benchmark/reporter.py`

- [ ] **Step 1: 创建 `reporter.py`**

```python
# src/benchmark/reporter.py
"""报告输出 — Rich 终端表格 + JSON 导出 + 历史趋势。"""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .benchmark_config import REPORT_DIR
from .database import BenchmarkDB
from .runner import BenchmarkReport, VersionResult


def print_report(report: BenchmarkReport, console: Console | None = None) -> None:
    """打印终端 benchmark 报告（版本对比表 + 趋势）。"""
    console = console or Console()

    console.print(Panel(
        f"[bold]Benchmark Report {report.run_id}[/bold]\n"
        f"Dataset: {report.dataset_name} | {report.question_count} questions\n"
        f"Commit: {report.git_commit}",
        border_style="cyan",
    ))

    # ---- 版本对比表 ----
    table = Table(title="版本对比")
    table.add_column("Metric", style="cyan")
    for vr in report.versions:
        table.add_column(vr.agent_version.upper(), justify="right")

    # 检索指标行
    r = report.retrieval
    if r:
        _add_metric_row(table, "Recall@5", [r.avg_recall.get(5, 0)] * len(report.versions), fmt=".1%")
        _add_metric_row(table, "MRR", [r.avg_mrr] * len(report.versions), fmt=".3f")
        _add_metric_row(table, "Hit@5", [r.avg_hit.get(5, 0)] * len(report.versions), fmt=".1%")
        _add_metric_row(table, "NDCG@5", [r.avg_ndcg.get(5, 0)] * len(report.versions), fmt=".3f")

    # 分隔线
    table.add_row("─" * 16, *(["─" * 10] * len(report.versions)))

    # RAGAS 指标行
    _add_metric_row(table, "Faithfulness", [vr.ragas_faithfulness for vr in report.versions], fmt=".1%")
    _add_metric_row(table, "Answer Relevance", [vr.ragas_answer_relevance for vr in report.versions], fmt=".1%")
    _add_metric_row(table, "Context Relevance", [vr.ragas_context_relevance for vr in report.versions], fmt=".1%")

    # 分隔线
    table.add_row("─" * 16, *(["─" * 10] * len(report.versions)))

    # 性能指标行
    _add_metric_row(table, "Avg TTFT (ms)", [vr.timing.avg_ttft_ms for vr in report.versions], fmt=".0f")
    _add_metric_row(table, "Avg Total (ms)", [vr.timing.avg_total_ms for vr in report.versions], fmt=".0f")

    # 分隔线
    table.add_row("─" * 16, *(["─" * 10] * len(report.versions)))

    # 通过状态行
    passed_vals = ["✓" if report.passed_map.get(vr.agent_version, False) else "✗ FAIL" for vr in report.versions]
    _add_metric_row(table, "Passed", passed_vals, fmt="")

    console.print(table)

    # ---- 趋势摘要 ----
    console.print("\n[bold]趋势: 本次 vs 上次[/bold]")
    db = BenchmarkDB()
    for vr in report.versions:
        last = db.get_last_run(vr.agent_version, report.dataset_name)
        trend = _build_trend_lines(vr, last, report.retrieval)
        console.print(f"  [cyan]{vr.agent_version}[/cyan]: {trend or '（首次运行，无历史数据）'}")

    console.print()


def _add_metric_row(table: Table, name: str, values: list, fmt: str = ".3f") -> None:
    """格式化一行指标到表格。"""
    formatted = []
    for v in values:
        if isinstance(v, str):
            formatted.append(v)
        elif fmt:
            formatted.append(f"{v:{fmt}}")
        else:
            formatted.append(str(v))
    table.add_row(name, *formatted)


def _build_trend_lines(vr: VersionResult, last_run: dict | None, retrieval) -> str:
    """构建趋势描述文字。"""
    if last_run is None:
        return ""
    lines = []
    # Faithfulness
    prev_f = last_run.get("faithfulness", 0)
    if prev_f:
        delta = vr.ragas_faithfulness - prev_f
        arrow = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "→"
        lines.append(f"Faithfulness: {vr.ragas_faithfulness:.1%} ({arrow}{abs(delta):.2f})")
    return ", ".join(lines)


def save_json_report(report: BenchmarkReport) -> str:
    """导出完整 JSON 报告到文件。"""
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{report.run_id}.json"

    data = {
        "run_id": report.run_id,
        "dataset_name": report.dataset_name,
        "question_count": report.question_count,
        "git_commit": report.git_commit,
        "total_elapsed_ms": report.total_elapsed_ms,
        "versions": {},
    }
    for vr in report.versions:
        data["versions"][vr.agent_version] = {
            "timing": {
                "avg_total_ms": vr.timing.avg_total_ms,
                "avg_ttft_ms": vr.timing.avg_ttft_ms,
                "avg_generation_ms": vr.timing.avg_generation_ms,
            },
            "ragas": {
                "faithfulness": vr.ragas_faithfulness,
                "answer_relevance": vr.ragas_answer_relevance,
                "context_relevance": vr.ragas_context_relevance,
            },
            "passed": report.passed_map.get(vr.agent_version, False),
        }

    if report.retrieval:
        data["retrieval"] = {
            "recall_at_5": report.retrieval.avg_recall.get(5, 0),
            "precision_at_5": report.retrieval.avg_precision.get(5, 0),
            "mrr": report.retrieval.avg_mrr,
            "ndcg_at_5": report.retrieval.avg_ndcg.get(5, 0),
            "hit_at_5": report.retrieval.avg_hit.get(5, 0),
        }

    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filepath)


def print_history(console: Console, db: BenchmarkDB, version: str = "", limit: int = 10) -> None:
    """打印历史趋势表格。"""
    rows = db.get_history(agent_version=version, limit=limit)
    if not rows:
        console.print("[dim]暂无历史记录[/dim]")
        return

    title = f"历史趋势 ({version or '全部版本'})"
    table = Table(title=title)
    table.add_column("Run ID", style="dim")
    table.add_column("Version", style="cyan")
    table.add_column("Recall@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("TTFT(ms)", justify="right")
    table.add_column("Pass", justify="center")

    for row in rows:
        passed = "✓" if row["passed"] else "✗"
        table.add_row(
            str(row["id"]),
            str(row["agent_version"]),
            f"{row['recall_at_5']:.1%}",
            f"{row['mrr']:.3f}",
            f"{row['faithfulness']:.1%}",
            f"{row['avg_ttft_ms']:.0f}",
            passed,
        )
    console.print(table)
```

- [ ] **Step 2: 验证语法**

```bash
uv run python -c "from src.benchmark.reporter import print_report, save_json_report, print_history; print('ok')"
```
预期: `ok`

- [ ] **Step 3: 提交**

```bash
git add src/benchmark/reporter.py
git commit -m "feat(benchmark): 新增报告输出模块

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: CLI 入口与 main.py 集成

**Files:**
- Create: `src/benchmark/cli.py`
- Modify: `src/main.py`

- [ ] **Step 1: 创建 `cli.py`**

```python
# src/benchmark/cli.py
"""Benchmark CLI 入口，供 main.py 调用。"""

import asyncio

from rich.console import Console

from .benchmark_config import DEFAULT_DATASET, DEFAULT_VERSIONS
from .database import BenchmarkDB
from .reporter import print_report, save_json_report, print_history
from .runner import BenchmarkRunner


async def run_benchmark(
    dataset: str = "",
    versions: str = "",
    profile_only: bool = False,
    eval_only: bool = False,
    set_baseline: bool = False,
    history: bool = False,
) -> None:
    """benchmark 命令分发入口。"""
    console = Console()

    if set_baseline:
        db = BenchmarkDB()
        db.init_default_baselines()
        console.print("[green]✓ 基线阈值已写入/更新[/green]")
        baselines = db.get_baselines()
        for name, cfg in baselines.items():
            console.print(f"  {name}: min={cfg['min_threshold']}, max_degrade={cfg['max_degradation_pct']}%")
        return

    if history:
        db = BenchmarkDB()
        version = versions.split(",")[0].strip() if versions else ""
        print_history(console, db, version=version)
        return

    dataset_path = dataset or DEFAULT_DATASET
    version_list = [v.strip() for v in versions.split(",") if v.strip()] if versions else DEFAULT_VERSIONS

    console.print(f"[bold cyan]=== Benchmark 开始 ===[/bold cyan]")
    console.print(f"数据集: {dataset_path}")
    console.print(f"版本: {', '.join(version_list)}")
    console.print(f"模式: {'仅性能' if profile_only else '仅评测' if eval_only else '完整'}\n")

    runner = BenchmarkRunner(
        dataset_path=dataset_path,
        versions=version_list,
        profile_only=profile_only,
        eval_only=eval_only,
    )
    report = await runner.run()

    print_report(report, console)
    json_path = save_json_report(report)
    console.print(f"[green]JSON 报告已保存: {json_path}[/green]")
```

- [ ] **Step 2: 修改 `src/main.py`，添加 benchmark 子命令**

在 `src/main.py` 的 `main()` 函数中，`eval_parser` 定义之后、`args = parser.parse_args()` 之前，添加 benchmark 子命令解析器：

```python
    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="RAG Benchmark 跨版本对比测试")
    bench_parser.add_argument(
        "--dataset", "-d",
        default="",
        help="评测数据集 JSON 文件路径（默认使用 generated_questions.json）",
    )
    bench_parser.add_argument(
        "--versions",
        default="",
        help="评测版本列表，逗号分隔（默认 v1,v2,v3,v4）",
    )
    bench_parser.add_argument(
        "--profile-only",
        action="store_true",
        help="仅性能压测，跳过质量评测",
    )
    bench_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="仅质量评测，跳过性能打点",
    )
    bench_parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="初始化/更新基线阈值",
    )
    bench_parser.add_argument(
        "--history",
        action="store_true",
        help="查看历史 benchmark 趋势",
    )
```

在 `main()` 函数末尾的 `elif` 链中，添加 `benchmark` 分支：

```python
    elif args.command == "benchmark":
        from .benchmark.cli import run_benchmark
        await run_benchmark(
            dataset=args.dataset,
            versions=args.versions,
            profile_only=args.profile_only,
            eval_only=args.eval_only,
            set_baseline=args.set_baseline,
            history=args.history,
        )
```

需要修改 `src/main.py` 的三处位置：
1. 在 eval_parser 之后添加 bench_parser（约第 262 行附近）
2. 在 `elif args.command == "eval":` 之后添加 benchmark 分支（约第 273 行附近）

- [ ] **Step 3: 验证 CLI 注册**

```bash
uv run python -m src.main benchmark --help
```
预期: 打印 benchmark 命令帮助信息

- [ ] **Step 4: 验证 `--set-baseline` 可用**

```bash
uv run python -m src.main benchmark --set-baseline
```
预期: 打印 6 条基线阈值

- [ ] **Step 5: 验证 `--history` 可用**

```bash
uv run python -m src.main benchmark --history
```
预期: 可能为空或显示刚写入的基线记录历史（实际为空因为还没跑过 benchmark）

- [ ] **Step 6: 提交**

```bash
git add src/benchmark/cli.py src/main.py
git commit -m "$(cat <<'EOF'
feat(benchmark): 新增 CLI 入口并集成到 main.py

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 端到端验证

**Files:** 无新文件

- [ ] **Step 1: 用小数据集快速验证完整流程**

```bash
uv run python -m src.main benchmark --profile-only --versions v1 -d eval_data/generated_questions.json
```

预期: 跑完 v1 的 30 条问题（仅性能打点），终端输出 benchmark 报告表格，`benchmark_results/` 目录下出现 JSON 报告和 SQLite 数据库。

- [ ] **Step 2: 验证 JSON 报告已写入**

```bash
ls benchmark_results/*.json | tail -1
```
预期: 输出刚生成的 JSON 文件路径

- [ ] **Step 3: 验证 SQLite 记录已写入**

```bash
uv run python -c "
from src.benchmark.database import BenchmarkDB
db = BenchmarkDB()
for r in db.get_history():
    print(r['id'], r['agent_version'], r['passed'])
"
```
预期: 打印刚跑的那条记录

- [ ] **Step 4: 验证历史查看命令**

```bash
uv run python -m src.main benchmark --history
```
预期: Rich 表格显示历史记录
