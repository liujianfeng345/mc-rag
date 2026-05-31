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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
                conn.commit()
        except Exception as e:
            raise RuntimeError(
                f"无法初始化 benchmark 数据库 schema（路径: {self.db_path}）：{e}"
            ) from e

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

        def _prepare_value(c: str):
            val = run_data.get(c, 0.0 if c != "report_json" else "{}")
            if c == "report_json" and isinstance(val, dict):
                return json.dumps(val)
            return val

        with self._connect() as conn:
            conn.execute(sql, tuple(_prepare_value(c) for c in columns))
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
