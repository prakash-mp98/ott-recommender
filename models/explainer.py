"""
Explainability engine — generates human-readable reasons for each recommendation.

Three signal sources:
  1. Social proof  — similar users who liked this item
  2. Genre match   — overlap between user's top genres and item's genres
  3. Popularity    — how well-rated the item is globally
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class Explainer:
    def __init__(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        cf_model,           # UserBasedCF instance
        top_genres_n: int = 3,
    ):
        self._ratings = ratings
        self._movies = movies.set_index("item_id")
        self._cf = cf_model
        self._top_genres_n = top_genres_n
        self._global_stats = self._compute_global_stats()

    def _compute_global_stats(self) -> pd.DataFrame:
        return (
            self._ratings.groupby("item_id")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_rating", "count": "n_ratings"})
        )

    def _user_top_genres(self, user_idx: int) -> List[str]:
        user_id = user_idx + 1
        user_ratings = self._ratings[
            (self._ratings["user_id"] == user_id) & (self._ratings["rating"] >= 4)
        ]
        genre_counts: Dict[str, int] = {}
        for item_id in user_ratings["item_id"]:
            if item_id in self._movies.index:
                for g in self._movies.loc[item_id, "genres"]:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        return [g for g, _ in sorted(genre_counts.items(), key=lambda x: -x[1])][: self._top_genres_n]

    def _item_genres(self, item_id: int) -> List[str]:
        if item_id not in self._movies.index:
            return []
        return self._movies.loc[item_id, "genres"]

    def explain(self, user_idx: int, item_idx: int) -> List[str]:
        item_id = item_idx + 1
        reasons: List[str] = []

        # 1. Social proof
        similar_who_liked = self._cf.users_who_liked(item_idx, user_idx)
        if similar_who_liked:
            reasons.append(
                f"{len(similar_who_liked)} viewers with your taste rated this ≥4★"
            )

        # 2. Genre match
        user_genres = set(self._user_top_genres(user_idx))
        item_genres = set(self._item_genres(item_id))
        overlap = user_genres & item_genres
        if overlap:
            genres_str = ", ".join(sorted(overlap)[:2])
            reasons.append(f"Matches your preference for {genres_str}")

        # 3. Global quality signal
        if item_id in self._global_stats.index:
            avg = self._global_stats.loc[item_id, "avg_rating"]
            count = int(self._global_stats.loc[item_id, "n_ratings"])
            if avg >= 4.0 and count >= 50:
                reasons.append(f"Highly rated: {avg:.1f}★ from {count} viewers")

        if not reasons:
            reasons.append("Recommended based on your watch history")

        return reasons

    def batch_explain(
        self, user_idx: int, item_indices: np.ndarray
    ) -> Dict[int, List[str]]:
        return {int(idx): self.explain(user_idx, int(idx)) for idx in item_indices}
