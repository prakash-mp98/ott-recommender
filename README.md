# OTT Recommender

> Hybrid movie recommender with explainability, diversity control, and cold-start handling built for OTT platforms.

[![CI](https://github.com/prakash-mp98/ott-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/prakash-mp98/ott-recommender/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Standard OTT recommenders rank purely by relevance and this one adds three things that real platforms actually struggle with:

| Problem | This project's solution |
|---|---|
| Black-box suggestions users don't trust | **Explainability engine** : human-readable reasons per recommendation |
| Genre bubble (watch one thriller, see only thrillers forever) | **MMR diversity re-ranking** : tunable balance between relevance and variety |
| New user with no history | **3-tier cold-start** : genre quiz → content-based → NCF as history builds |

Dataset: **MovieLens 100K** (943 users · 1,682 movies · 100,000 ratings)

---

## Architecture

```
Request
   │
   ▼
┌─────────────────────────────────────────┐
│              Mode selector              │
│  ≥20 ratings → NCF                     │
│   1–19       → Content-based (history) │
│   0          → Genre cold-start        │
└──────────────┬──────────────────────────┘
               │  candidate pool (n×5)
               ▼
     MMR Diversity Re-ranker
     λ controls relevance ↔ diversity
               │  top-N diverse candidates
               ▼
     Explainability Engine
     ├─ Social proof (similar users who liked it)
     ├─ Genre match (overlap with your taste profile)
     └─ Quality signal (global avg rating)
               │
               ▼
           Response JSON
```

### Models

| Model | Role |
|---|---|
| **NeuMF (NCF)** | GMF + MLP neural collaborative filtering — main ranker for warm users |
| **UserBasedCF** | Cosine-similarity CF — powers the explainability social-proof signal |
| **ContentBasedRecommender** | Genre one-hot + TF-IDF on titles — cold-start and similar-movie lookups |
| **MMR Re-ranker** | Maximal Marginal Relevance diversity post-processing |
| **Explainer** | Generates per-item reasons from CF + genre overlap + global stats |

---

## Quick Start

```bash
git clone https://github.com/prakash-mp98/ott-recommender.git
cd ott-recommender
docker compose up --build
```

MovieLens 100K is downloaded automatically on first run.
API: `http://localhost:8000` · Swagger UI: `http://localhost:8000/docs`

---

## API Endpoints

### Personalised recommendations

```bash
curl "http://localhost:8000/recommend/1?n=5&diversity=0.4&explain=true"
```

```json
{
  "user_id": 1,
  "mode": "ncf+diversity",
  "diversity_lambda": 0.4,
  "recommendations": [
    {
      "item_id": 50,
      "title": "Star Wars (1977)",
      "year": 1977,
      "genres": ["Action", "Adventure", "Sci-Fi"],
      "score": 0.9821,
      "reasons": [
        "12 viewers with your taste rated this ≥4★",
        "Matches your preference for Action, Sci-Fi",
        "Highly rated: 4.4★ from 584 viewers"
      ]
    }
  ]
}
```

### Cold-start (brand new user)

```bash
curl "http://localhost:8000/cold-start?genres=Action,Sci-Fi&n=5&diversity=0.5"
```

### Similar movies

```bash
curl "http://localhost:8000/similar/1?n=5"
```

### Browse catalogue

```bash
curl "http://localhost:8000/movies?genre=Drama&limit=10"
```

---

## The Novelty

**Most tutorial recommenders stop at CF or NCF.** This project adds three production-relevant layers:

**1  Explainability**
Each recommendation comes with human-readable reasons derived from:
- Social proof: how many similar users liked it
- Taste match: genre overlap with the user's watch history
- Quality: global average rating and review count

**2 Diversity (MMR)**
Maximal Marginal Relevance re-ranks candidates by:
```
score(i) = λ × relevance(i) − (1−λ) × max_similarity_to_already_selected(i)
```
`diversity=0.0` → pure relevance · `diversity=1.0` → maximum variety.
This directly addresses the filter-bubble problem that Netflix and Prime Video face.

**3  Cold-start tiers**
| User state | Mode |
|---|---|
| 0 ratings | Genre preference → content-based |
| 1–19 ratings | Liked items → content-based profile |
| ≥20 ratings | NCF + MMR |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Precision@K | Fraction of top-K that are relevant (rating ≥ 4★) |
| Recall@K | Fraction of all relevant items captured in top-K |
| NDCG@K | Ranking quality -> rewards placing best items first |
| Intra-list diversity | Average pairwise cosine distance in the recommendation list |

---

## Local Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v            # 14 tests, no download needed
uvicorn api.main:app --reload
```

---

## Project Structure

```
ott-recommender/
├── data/
│   └── loader.py            # MovieLens 100K download + parsing
├── models/
│   ├── ncf.py               # NeuMF — GMF + MLP neural CF
│   ├── collaborative.py     # User-based CF (explainability backbone)
│   ├── content.py           # Genre + TF-IDF content-based (cold-start)
│   ├── diversity.py         # MMR diversity re-ranker
│   └── explainer.py         # Human-readable reason generator
├── evaluation/
│   └── metrics.py           # Precision@K, Recall@K, NDCG@K, diversity
├── api/
│   └── main.py              # FastAPI — 4 endpoints, 3-tier mode selector
├── tests/
│   └── test_recommender.py  # 14 unit tests (synthetic data, no download)
├── Dockerfile
└── docker-compose.yml
```

---

## License

MIT © [Prakash](https://github.com/prakash-mp98)
