"""RAG 系统评测模块

提供三层评测能力：
1. 检索质量评测：Recall@K, Precision@K, MRR, NDCG@K
2. 生成质量评测：Faithfulness, Answer Relevance, Context Relevance（LLM 裁判）
3. 版本对比评测：v1-v4 多版本横向对比

使用方式：
    uv run python -m src.main eval --dataset eval_data/sample_questions.json
    uv run python -m src.main eval --dataset eval_data/sample_questions.json --retrieval-only
    uv run python -m src.main eval --dataset eval_data/sample_questions.json --compare-versions
"""
