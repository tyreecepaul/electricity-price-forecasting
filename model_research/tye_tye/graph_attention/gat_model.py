"""Plain, spatial-only GNN model for joint 5-region NEM price forecasting.

Deliberately has no Transformer/LSTM temporal block: a 2-layer GAT is applied
independently at every timestep, then the sequence is reduced with mean
pooling over time. This isolates how much the graph/spatial structure alone
buys before adding any temporal-attention component in a later pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_attention_layer import GraphAttentionLayer
from graph_data import NEM_ADJACENCY
from evaluation.Model import Model


class GATNetwork(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        adjacency = torch.tensor(NEM_ADJACENCY, dtype=torch.float32)

        self.gat1 = GraphAttentionLayer(n_features, hidden_dim, adjacency, num_heads=num_heads)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, adjacency, num_heads=num_heads)
        self.activation = nn.GELU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, num_nodes, n_features]
        batch_size, seq_len, num_nodes, n_features = x.shape
        folded = x.reshape(batch_size * seq_len, num_nodes, n_features)

        hidden = self.activation(self.gat1(folded))
        hidden = self.activation(self.gat2(hidden))

        hidden = hidden.view(batch_size, seq_len, num_nodes, -1)
        pooled = hidden.mean(dim=1)  # [batch, num_nodes, hidden_dim]

        return self.head(pooled).squeeze(-1)  # [batch, num_nodes]


class GATModel(Model):
    model_name = "GAT (spatial-only)"

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network = GATNetwork(n_features, hidden_dim=hidden_dim, num_heads=num_heads).to(
            self.device
        )
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, training_data) -> None:
        X = torch.tensor(training_data.X, dtype=torch.float32, device=self.device)
        y = torch.tensor(training_data.y, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.network.train()
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                predictions = self.network(X[idx])
                loss = loss_fn(predictions, y[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            print(f"epoch {epoch + 1}/{self.epochs} loss={epoch_loss / n_samples:.4f}")

    def predict(self, test_data) -> np.ndarray:
        self.network.eval()
        X = torch.tensor(test_data.X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self.network(X)
        return predictions.cpu().numpy()

    def get_model_name(self) -> str:
        return self.model_name
