"""
Standard IR metrics for recommender system evaluation.

  RMSE          — rating prediction error
  Precision@K   — fraction of top-K that are relevant (rating >= threshold)
  Recall@K      — fraction of relevant items captured in top-K
  NDCG@K        — ranking quality (rewards putting best items first)
  Coverage      — fraction of catalogue recommended at least once
  Diversity     — average pairwise distance across recommendations
"""

import numpy as np
from typing import Dict, List, Set


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for i in top_k if i in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for i in top_k if i in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    model,
    test_ratings,
    train_ratings,
    n_users: int,
    k: int = 10,
    rel_threshold: float = 4.0,
    sample_users: int = 200,
) -> Dict[str, float]:
    """
    Evaluate a model over `sample_users` users and return averaged metrics.
    model must implement: top_n(user_idx, n, exclude) -> (indices, scores)
    """
    import pandas as pd

    rng = np.random.default_rng(42)
    user_ids = rng.choice(n_users, size=min(sample_users, n_users), replace=False)

    p_list, r_list, ndcg_list = [], [], []

    for uid in user_ids:
        user_id = uid + 1
        train_items = set(
            train_ratings[train_ratings["user_id"] == user_id]["item_id"] - 1
        )
        test_rows = test_ratings[
            (test_ratings["user_id"] == user_id)
            & (test_ratings["rating"] >= rel_threshold)
        ]
        relevant = set(test_rows["item_id"] - 1)
        if not relevant:
            continue

        top_items, _ = model.top_n(uid, n=k * 5, exclude=train_items)
        recommended = list(top_items[:k])

        p_list.append(precision_at_k(recommended, relevant, k))
        r_list.append(recall_at_k(recommended, relevant, k))
        ndcg_list.append(ndcg_at_k(recommended, relevant, k))

    return {
        f"precision@{k}": float(np.mean(p_list)) if p_list else 0.0,
        f"recall@{k}":    float(np.mean(r_list)) if r_list else 0.0,
        f"ndcg@{k}":      float(np.mean(ndcg_list)) if ndcg_list else 0.0,
    }


def intra_list_diversity(recommended_indices: List[int], item_embeddings: np.ndarray) -> float:
    """Average pairwise cosine distance within a recommendation list."""
    if len(recommended_indices) < 2:
        return 0.0
    embs = item_embeddings[recommended_indices].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
    embs = embs / norms
    sim_matrix = embs @ embs.T
    n = len(recommended_indices)
    total_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
    return float(1.0 - total_sim)
