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
