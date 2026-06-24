"""
Content-based recommender using genre vectors + TF-IDF on titles.
Used as the cold-start fallback when a user has little or no history.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple

from data.loader import GENRE_LABELS


class ContentBasedRecommender:
    def __init__(self, genre_weight: float = 0.7):
        self.genre_weight = genre_weight
        self._genre_matrix: np.ndarray = None
        self._tfidf_matrix: np.ndarray = None
        self._item_matrix: np.ndarray = None
        self._movies: pd.DataFrame = None

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        self._movies = movies.reset_index(drop=True)

        # Genre one-hot vectors
        genre_vecs = np.zeros((len(movies), len(GENRE_LABELS)), dtype=np.float32)
        for i, genres in enumerate(movies["genres"]):
            for g in genres:
                if g in GENRE_LABELS:
                    genre_vecs[i, GENRE_LABELS.index(g)] = 1.0
        # L2-normalize
        norms = np.linalg.norm(genre_vecs, axis=1, keepdims=True).clip(min=1e-8)
        self._genre_matrix = genre_vecs / norms

        # TF-IDF on titles
        tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=2000)
        self._tfidf_matrix = tfidf.fit_transform(movies["title"].fillna("")).toarray()

        # Concatenate weighted genre + TF-IDF vectors into a single feature space
        self._item_matrix = np.concatenate(
            [
                self.genre_weight * self._genre_matrix,
                (1 - self.genre_weight) * self._tfidf_matrix,
            ],
            axis=1,
        )
        return self

    def similar_items(self, item_idx: int, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        sims = cosine_similarity(
            self._item_matrix[item_idx].reshape(1, -1), self._item_matrix
        )[0]
        sims[item_idx] = -1.0
        top_idx = np.argsort(sims)[::-1][:n]
        return top_idx, sims[top_idx]

    def recommend_from_genres(
        self, preferred_genres: List[str], n: int = 10, exclude: set = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cold-start: user tells us their genre preferences."""
        # Build query in genre space, pad zeros for TF-IDF dimensions
        genre_query = np.zeros(len(GENRE_LABELS), dtype=np.float32)
        for g in preferred_genres:
            if g in GENRE_LABELS:
                genre_query[GENRE_LABELS.index(g)] = 1.0
        norm = np.linalg.norm(genre_query)
        if norm > 0:
            genre_query /= norm
        tfidf_dim = self._item_matrix.shape[1] - len(GENRE_LABELS)
        query = np.concatenate([self.genre_weight * genre_query,
                                np.zeros(tfidf_dim, dtype=np.float32)])
        sims = self._item_matrix @ query
        if exclude:
            sims[list(exclude)] = -np.inf
        top_idx = np.argsort(sims)[::-1][:n]
        return top_idx, sims[top_idx]

    def recommend_from_history(
        self, liked_item_indices: List[int], n: int = 10, exclude: set = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Partial cold-start: user has a few ratings, build a taste profile."""
        if not liked_item_indices:
            return np.array([]), np.array([])
        profile = self._item_matrix[liked_item_indices].mean(axis=0)
        sims = self._item_matrix @ profile
        if exclude:
            sims[list(exclude)] = -np.inf
        top_idx = np.argsort(sims)[::-1][:n]
        return top_idx, sims[top_idx]

    def item_vectors(self) -> np.ndarray:
        return self._item_matrix
