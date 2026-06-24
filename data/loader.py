"""
MovieLens 100K data loader.
Downloads and parses ratings + movie metadata on first use.
"""

import os
import urllib.request
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "ml-100k")

GENRE_LABELS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download(data_dir: str = _DATA_DIR) -> str:
    if os.path.exists(os.path.join(data_dir, "u.data")):
        return data_dir
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    zip_path = data_dir + ".zip"
    print("Downloading MovieLens 100K …")
    urllib.request.urlretrieve(_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(os.path.dirname(data_dir))
    os.remove(zip_path)
    return data_dir


def load_ratings(data_dir: str = _DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, "u.data")
    df = pd.read_csv(
        path, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # 0-indexed IDs for embedding lookup
    df["user_idx"] = df["user_id"] - 1
    df["item_idx"] = df["item_id"] - 1
    return df


def load_movies(data_dir: str = _DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, "u.item")
    cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRE_LABELS
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")
    df = df[["item_id", "title", "release_date"] + GENRE_LABELS].copy()
    df["genres"] = df[GENRE_LABELS].apply(
        lambda row: [g for g, v in zip(GENRE_LABELS, row) if v == 1], axis=1
    )
    df["year"] = df["release_date"].str.extract(r"(\d{4})").astype("float")
    return df[["item_id", "title", "year", "genres"]]


def train_test_split(
    ratings: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_mask = rng.random(len(ratings)) < test_ratio
    return ratings[~test_mask].reset_index(drop=True), ratings[test_mask].reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    return csr_matrix(
        (ratings["rating"].values, (ratings["user_idx"].values, ratings["item_idx"].values)),
        shape=(n_users, n_items),
    )
