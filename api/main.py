"""
OTT Recommender API — FastAPI control plane.

Three recommendation modes selected automatically:
  • Warm user  (≥20 ratings)  → NCF + MMR diversity re-ranking + explanations
  • Cold user  (1–19 ratings) → Content-based from liked items + MMR
  • New user   (0 ratings)    → Genre preference query → content-based
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.loader import download, load_movies, load_ratings, train_test_split, build_user_item_matrix
from models.ncf import NCFRecommender
from models.collaborative import UserBasedCF
from models.content import ContentBasedRecommender
from models.diversity import mmr_rerank
from models.explainer import Explainer

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_state: Dict = {}

WARM_THRESHOLD = 20   # ratings needed to use NCF
COLD_THRESHOLD  = 1   # ratings needed for content-based fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading MovieLens 100K …")
    download()
    ratings   = load_ratings()
    movies    = load_movies()
    train, _  = train_test_split(ratings)

    n_users = int(ratings["user_idx"].max()) + 1
    n_items = int(ratings["item_idx"].max()) + 1

    ui_matrix = build_user_item_matrix(train, n_users, n_items)

    # Positive pairs for NCF (rating >= 4 = implicit positive)
    pos_pairs = list(
        train[train["rating"] >= 4][["user_idx", "item_idx"]].itertuples(index=False, name=None)
    )

    print("Training UserCF …")
    cf = UserBasedCF(n_similar=30).fit(ui_matrix)

    print("Training NCF …")
    ncf = NCFRecommender(n_users, n_items, emb_dim=32, epochs=10).fit(pos_pairs)

    print("Fitting content-based model …")
    cb = ContentBasedRecommender().fit(movies)

    explainer = Explainer(ratings=train, movies=movies, cf_model=cf)

    _state.update(
        ratings=ratings,
        train=train,
        movies=movies,
        n_users=n_users,
        n_items=n_items,
        cf=cf,
        ncf=ncf,
        cb=cb,
        explainer=explainer,
    )
    print("All models ready.")
    yield
    _state.clear()


app = FastAPI(
    title="OTT Recommender API",
    description=(
        "Hybrid movie recommender: **NCF** ranking + **MMR diversity** re-ranking "
        "+ **explainability** + **cold-start** handling.\n\n"
        "Dataset: MovieLens 100K (943 users · 1,682 movies · 100,000 ratings)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RecommendedItem(BaseModel):
    item_id: int
    title: str
    year: Optional[float]
    genres: List[str]
    score: float
    reasons: List[str]


class RecommendResponse(BaseModel):
    user_id: int
    mode: str
    diversity_lambda: float
    recommendations: List[RecommendedItem]


class SimilarItem(BaseModel):
    item_id: int
    title: str
    genres: List[str]
    similarity: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _movie_info(item_idx: int) -> dict:
    movies = _state["movies"]
    row = movies[movies["item_id"] == item_idx + 1]
    if row.empty:
        return {"item_id": item_idx + 1, "title": "Unknown", "year": None, "genres": []}
    r = row.iloc[0]
    return {
        "item_id": int(r["item_id"]),
        "title": str(r["title"]),
        "year": float(r["year"]) if r["year"] == r["year"] else None,
        "genres": list(r["genres"]),
    }


def _user_history(user_idx: int):
    ratings = _state["ratings"]
    user_id = user_idx + 1
    rows = ratings[ratings["user_id"] == user_id]
    return rows


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health():
    return {
        "status": "ok",
        "n_users": _state.get("n_users"),
        "n_items": _state.get("n_items"),
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(
    user_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    diversity: float = Query(0.5, ge=0.0, le=1.0, description="0=relevant, 1=diverse"),
    explain: bool = Query(True, description="Include human-readable reasons"),
):
    """
    Get personalised movie recommendations for a user.

    **Modes selected automatically:**
    - Warm user (≥20 ratings) → NCF + MMR diversity
    - Cold user (1–19 ratings) → Content-based from watch history + MMR
    - New user (0 ratings) → Popularity-based fallback
    """
    n_users = _state["n_users"]
    user_idx = user_id - 1

    if user_idx < 0 or user_idx >= n_users:
        raise HTTPException(status_code=404, detail=f"user_id must be 1–{n_users}")

    history = _user_history(user_idx)
    n_rated = len(history)
    watched = set(history["item_idx"].tolist())

    ncf: NCFRecommender   = _state["ncf"]
    cb: ContentBasedRecommender = _state["cb"]
    explainer: Explainer  = _state["explainer"]

    # ---- select mode ----
    if n_rated >= WARM_THRESHOLD:
        mode = "ncf+diversity"
        candidate_idx, candidate_scores = ncf.top_n(user_idx, n=n * 5, exclude=watched)
        embs = ncf.item_embeddings()

    elif n_rated >= COLD_THRESHOLD:
        mode = "content+diversity"
        liked = list(history[history["rating"] >= 4]["item_idx"])
        candidate_idx, candidate_scores = cb.recommend_from_history(
            liked, n=n * 5, exclude=watched
        )
        embs = cb.item_vectors()

    else:
        mode = "popularity"
        ratings = _state["ratings"]
        popular = (
            ratings.groupby("item_idx")["rating"]
            .agg(["mean", "count"])
            .query("count >= 50")
            .sort_values("mean", ascending=False)
            .head(n * 5)
        )
        candidate_idx = popular.index.to_numpy()
        candidate_scores = popular["mean"].to_numpy()
        embs = cb.item_vectors()

    # ---- MMR re-rank ----
    final_idx, final_scores = mmr_rerank(
        candidate_idx, candidate_scores, embs, k=n, lambda_=1.0 - diversity
    )

    # ---- build response ----
    reasons_map = (
        explainer.batch_explain(user_idx, final_idx) if explain else {}
    )

    recs = []
    for idx, score in zip(final_idx, final_scores):
        info = _movie_info(int(idx))
        recs.append(
            RecommendedItem(
                **info,
                score=round(float(score), 4),
                reasons=reasons_map.get(int(idx), []),
            )
        )

    return RecommendResponse(
        user_id=user_id,
        mode=mode,
        diversity_lambda=diversity,
        recommendations=recs,
    )


@app.get("/similar/{movie_id}", response_model=List[SimilarItem], tags=["Recommendations"])
async def similar_movies(
    movie_id: int,
    n: int = Query(10, ge=1, le=30),
):
    """Find movies similar to a given movie using content-based similarity."""
    n_items = _state["n_items"]
    item_idx = movie_id - 1
    if item_idx < 0 or item_idx >= n_items:
        raise HTTPException(status_code=404, detail=f"movie_id must be 1–{n_items}")

    cb: ContentBasedRecommender = _state["cb"]
    top_idx, sims = cb.similar_items(item_idx, n=n)

    result = []
    for idx, sim in zip(top_idx, sims):
        info = _movie_info(int(idx))
        result.append(SimilarItem(**info, similarity=round(float(sim), 4)))
    return result


@app.get("/cold-start", response_model=RecommendResponse, tags=["Recommendations"])
async def cold_start(
    genres: str = Query(..., description="Comma-separated genre list, e.g. Action,Comedy"),
    n: int = Query(10, ge=1, le=50),
    diversity: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Recommend movies for a **brand-new user** based only on genre preferences.

    No user_id or watch history needed.
    """
    preferred = [g.strip() for g in genres.split(",") if g.strip()]
    if not preferred:
        raise HTTPException(status_code=422, detail="Provide at least one genre.")

    cb: ContentBasedRecommender = _state["cb"]
    candidate_idx, candidate_scores = cb.recommend_from_genres(preferred, n=n * 5)
    final_idx, final_scores = mmr_rerank(
        candidate_idx, candidate_scores, cb.item_vectors(), k=n, lambda_=1.0 - diversity
    )

    recs = []
    for idx, score in zip(final_idx, final_scores):
        info = _movie_info(int(idx))
        recs.append(RecommendedItem(**info, score=round(float(score), 4), reasons=[f"Matches your genre: {', '.join(preferred[:2])}"]))

    return RecommendResponse(
        user_id=0,
        mode="cold-start-genre",
        diversity_lambda=diversity,
        recommendations=recs,
    )


@app.get("/movies", tags=["Catalogue"])
async def list_movies(
    genre: Optional[str] = Query(None, description="Filter by genre"),
    limit: int = Query(20, ge=1, le=200),
):
    """Browse the movie catalogue, optionally filtered by genre."""
    movies = _state["movies"]
    if genre:
        movies = movies[movies["genres"].apply(lambda gs: genre in gs)]
    subset = movies.head(limit)
    return subset[["item_id", "title", "year", "genres"]].to_dict(orient="records")
