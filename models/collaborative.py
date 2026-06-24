"""
User-based Collaborative Filtering.
Used both as a standalone recommender and as the backbone for the explainer
(finding similar users who liked a given item).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class UserBasedCF:
    def __init__(self, n_similar: int = 20):
        self.n_similar = n_similar
        self._matrix: np.ndarray = None     # dense user-item
        self._sim: np.ndarray = None        # user-user similarity

    def fit(self, ui_matrix: csr_matrix) -> "UserBasedCF":
        self._matrix = ui_matrix.toarray().astype(np.float32)
        # Mean-center each user's ratings (only non-zero entries)
        means = np.true_divide(
            self._matrix.sum(axis=1),
            (self._matrix != 0).sum(axis=1).clip(min=1),
        )
        centered = self._matrix.copy()
        mask = centered != 0
        centered[mask] -= means[np.where(mask)[0]]
        self._sim = cosine_similarity(centered)
        np.fill_diagonal(self._sim, 0)
        return self

    def similar_users(self, user_idx: int, n: int = None) -> List[int]:
        n = n or self.n_similar
        sims = self._sim[user_idx]
        return list(np.argsort(sims)[::-1][:n])

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        sims = self._sim[user_idx]
        neighbors = np.argsort(sims)[::-1][: self.n_similar]
        rated_mask = self._matrix[neighbors, item_idx] != 0
        neighbors = neighbors[rated_mask]
        if len(neighbors) == 0:
            return 0.0
        weights = sims[neighbors]
        ratings = self._matrix[neighbors, item_idx]
        denom = np.abs(weights).sum()
        return float(np.dot(weights, ratings) / denom) if denom else 0.0

    def users_who_liked(self, item_idx: int, user_idx: int, min_rating: float = 4.0) -> List[int]:
        """Similar users to `user_idx` who rated `item_idx` >= min_rating."""
        similar = self.similar_users(user_idx)
        return [u for u in similar if self._matrix[u, item_idx] >= min_rating]

    def top_n(self, user_idx: int, n: int = 10, exclude: set = None) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.array(
            [self.predict_rating(user_idx, i) for i in range(self._matrix.shape[1])]
        )
        if exclude:
            scores[list(exclude)] = -np.inf
        top_idx = np.argsort(scores)[::-1][:n]
        return top_idx, scores[top_idx]
