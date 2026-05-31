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
