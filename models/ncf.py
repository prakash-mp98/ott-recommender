"""
NeuMF — Neural Matrix Factorization (He et al., 2017).
Combines Generalized Matrix Factorization (GMF) + MLP in one model.
Trained on implicit feedback: rating >= 4 → positive signal.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    def __init__(
        self,
        pos_pairs: List[Tuple[int, int]],
        n_items: int,
        n_neg: int = 4,
    ):
        self.n_items = n_items
        self.n_neg = n_neg
        self.pos_set = set(pos_pairs)
        self.data: List[Tuple[int, int, float]] = []

        for u, i in pos_pairs:
            self.data.append((u, i, 1.0))
            for _ in range(n_neg):
                j = np.random.randint(n_items)
                while (u, j) in self.pos_set:
                    j = np.random.randint(n_items)
                self.data.append((u, j, 0.0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        u, i, label = self.data[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        mlp_layers: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        mlp_layers = mlp_layers or [64, 32, 16]

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # MLP tower
        layers, in_dim = [], emb_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output
        self.out = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_in = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        return torch.sigmoid(self.out(torch.cat([gmf, mlp_out], dim=-1))).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NCFRecommender:
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        mlp_layers: List[int] = None,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.device = torch.device(device)
        self.model = NeuMF(n_users, n_items, emb_dim, mlp_layers or [64, 32, 16]).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._item_embs: np.ndarray = None  # cached after training

    def fit(self, pos_pairs: List[Tuple[int, int]]) -> "NCFRecommender":
        dataset = InteractionDataset(pos_pairs, self.n_items)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for u, i, label in loader:
                u, i, label = u.to(self.device), i.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(u, i), label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 2 == 0 or epoch == self.epochs:
                print(f"  [NCF] epoch {epoch}/{self.epochs}  loss={total_loss/len(loader):.4f}")

        self._cache_item_embeddings()
        return self

    def _cache_item_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            ids = torch.arange(self.n_items, device=self.device)
            gmf = self.model.gmf_item(ids).cpu().numpy()
            mlp = self.model.mlp_item(ids).cpu().numpy()
            self._item_embs = np.concatenate([gmf, mlp], axis=1)

    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([user_idx] * len(item_indices), device=self.device)
            i = torch.tensor(item_indices, device=self.device)
            return self.model(u, i).cpu().numpy()

    def item_embeddings(self) -> np.ndarray:
        return self._item_embs

    def top_n(self, user_idx: int, n: int = 10, exclude: set = None) -> np.ndarray:
        all_items = np.arange(self.n_items)
        if exclude:
            all_items = all_items[~np.isin(all_items, list(exclude))]
        scores = self.score(user_idx, all_items)
        top_idx = np.argsort(scores)[::-1][:n]
        return all_items[top_idx], scores[top_idx]
