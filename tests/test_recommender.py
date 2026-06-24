"""
Unit tests — use a small synthetic dataset (no MovieLens download needed).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

N_USERS, N_ITEMS = 30, 50

@pytest.fixture(scope="module")
def ratings_df():
    rng = np.random.default_rng(0)
    rows = {
        "user_id":  rng.integers(1, N_USERS + 1, 400),
        "item_id":  rng.integers(1, N_ITEMS + 1, 400),
        "rating":   rng.integers(1, 6, 400).astype(float),
        "timestamp": np.zeros(400, dtype=int),
    }
    df = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"])
    df["user_idx"] = df["user_id"] - 1
    df["item_idx"] = df["item_id"] - 1
    return df


@pytest.fixture(scope="module")
def movies_df():
    titles = [f"Movie {i}" for i in range(1, N_ITEMS + 1)]
    genres_pool = [["Action", "Drama"], ["Comedy"], ["Sci-Fi", "Action"], ["Romance"], ["Horror"]]
    return pd.DataFrame({
        "item_id": list(range(1, N_ITEMS + 1)),
        "title":   titles,
        "year":    [2000 + i % 24 for i in range(N_ITEMS)],
        "genres":  [genres_pool[i % len(genres_pool)] for i in range(N_ITEMS)],
    })


@pytest.fixture(scope="module")
def ui_matrix(ratings_df):
    return csr_matrix(
        (ratings_df["rating"].values, (ratings_df["user_idx"].values, ratings_df["item_idx"].values)),
        shape=(N_USERS, N_ITEMS),
    )


# ---------------------------------------------------------------------------
# UserBasedCF
# ---------------------------------------------------------------------------

from models.collaborative import UserBasedCF

def test_cf_fit(ui_matrix):
    cf = UserBasedCF().fit(ui_matrix)
    assert cf._sim.shape == (N_USERS, N_USERS)
    assert np.allclose(np.diag(cf._sim), 0)  # self-similarity zeroed

def test_cf_similar_users(ui_matrix):
    cf = UserBasedCF(n_similar=5).fit(ui_matrix)
    sim = cf.similar_users(0, n=5)
    assert len(sim) == 5
    assert 0 not in sim

def test_cf_top_n(ui_matrix):
    cf = UserBasedCF().fit(ui_matrix)
    idx, scores = cf.top_n(0, n=10)
    assert len(idx) == 10
    assert (np.diff(scores) <= 1e-6).all()  # descending


# ---------------------------------------------------------------------------
# ContentBasedRecommender
# ---------------------------------------------------------------------------

from models.content import ContentBasedRecommender

def test_cb_fit(movies_df):
    cb = ContentBasedRecommender().fit(movies_df)
    assert cb._item_matrix.shape[0] == N_ITEMS

def test_cb_similar_items(movies_df):
    cb = ContentBasedRecommender().fit(movies_df)
    idx, sims = cb.similar_items(0, n=5)
    assert len(idx) == 5
    assert 0 not in idx  # self excluded

def test_cb_cold_start_genre(movies_df):
    cb = ContentBasedRecommender().fit(movies_df)
    idx, sims = cb.recommend_from_genres(["Action"], n=5)
    assert len(idx) == 5

def test_cb_cold_start_history(movies_df):
    cb = ContentBasedRecommender().fit(movies_df)
    idx, sims = cb.recommend_from_history([0, 2, 4], n=5, exclude={0, 2, 4})
    assert len(idx) == 5
    assert not ({0, 2, 4} & set(idx.tolist()))


# ---------------------------------------------------------------------------
# MMR Diversity
# ---------------------------------------------------------------------------

from models.diversity import mmr_rerank

def test_mmr_returns_k():
    embs = np.random.rand(20, 16).astype(np.float32)
    scores = np.random.rand(20).astype(np.float32)
    candidates = np.arange(20)
    sel, _ = mmr_rerank(candidates, scores, embs, k=5, lambda_=0.5)
    assert len(sel) == 5

def test_mmr_diversity_vs_relevance():
    embs = np.eye(10, dtype=np.float32)  # perfectly orthogonal items
    scores = np.linspace(1, 0, 10).astype(np.float32)
    candidates = np.arange(10)
    # Pure relevance → always picks highest scores in order
    sel_rel, _ = mmr_rerank(candidates, scores, embs, k=5, lambda_=1.0)
    assert list(sel_rel) == [0, 1, 2, 3, 4]
    # Pure diversity on orthogonal → all selections equally diverse
    sel_div, _ = mmr_rerank(candidates, scores, embs, k=5, lambda_=0.0)
    assert len(set(sel_div)) == 5  # 5 unique items


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, intra_list_diversity

def test_precision():
    assert precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5) == pytest.approx(3 / 5)

def test_recall():
    assert recall_at_k([1, 2, 3], {1, 3, 7}, k=3) == pytest.approx(2 / 3)

def test_ndcg_perfect():
    # Perfect ranking: all relevant items at top
    relevant = {0, 1, 2}
    assert ndcg_at_k([0, 1, 2, 3, 4], relevant, k=5) == pytest.approx(1.0)

def test_ndcg_zero():
    assert ndcg_at_k([3, 4, 5], {0, 1, 2}, k=3) == pytest.approx(0.0)

def test_diversity():
    embs = np.eye(5, dtype=np.float32)
    score = intra_list_diversity([0, 1, 2, 3, 4], embs)
    assert score == pytest.approx(1.0)  # fully orthogonal → max diversity
