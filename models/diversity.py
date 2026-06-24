"""
Maximal Marginal Relevance (MMR) diversity re-ranker.

Standard recommenders rank purely by relevance → users get stuck in a genre
bubble. MMR balances relevance against redundancy with already-selected items.

  MMR score = λ × relevance(i) − (1−λ) × max_{j∈selected} sim(i, j)

λ = 1.0 → pure relevance (no diversity)
λ = 0.0 → pure diversity (no relevance)
λ = 0.5 → balanced (default)
"""

import numpy as np
from typing import List, Tuple


def mmr_rerank(
    candidate_indices: np.ndarray,
    relevance_scores: np.ndarray,
    item_embeddings: np.ndarray,
    k: int = 10,
    lambda_: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    candidate_indices : shape (N,)  — item indices from the full catalogue
    relevance_scores  : shape (N,)  — model scores for each candidate
    item_embeddings   : shape (total_items, dim) — full embedding table
    k                 : number of items to return
    lambda_           : diversity trade-off (0 = diverse, 1 = relevant)

    Returns
    -------
    selected_indices, selected_scores
    """
    if len(candidate_indices) == 0:
        return np.array([]), np.array([])

    k = min(k, len(candidate_indices))

    # Normalise embeddings for cosine similarity
    embs = item_embeddings[candidate_indices].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
    embs_normed = embs / norms

    selected: List[int] = []      # indices into candidate_indices
    remaining = list(range(len(candidate_indices)))

    for _ in range(k):
        if not remaining:
            break

        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda idx: relevance_scores[idx])
        else:
            sel_embs = embs_normed[selected]

            def _mmr(idx: int) -> float:
                rel = float(relevance_scores[idx])
                sims = embs_normed[idx] @ sel_embs.T
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                return lambda_ * rel - (1 - lambda_) * max_sim

            best = max(remaining, key=_mmr)

        selected.append(best)
        remaining.remove(best)

    sel = np.array(selected)
    return candidate_indices[sel], relevance_scores[sel]
