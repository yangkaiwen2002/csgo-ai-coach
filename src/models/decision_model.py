"""
Decision Model
--------------
Transformer-based model that predicts the optimal CS:GO action
given a sequence of game state features.

Training modes (as suggested by domain expert):
  1. Supervised    — labels from pro demo win-rate (high win rate action = correct)
  2. Semi-supervised — small labeled set + large unlabeled set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional
import yaml


ACTIONS = ["push_A", "push_B", "hold_A", "hold_B", "fall_back",
           "rotate_A", "rotate_B", "smoke_A", "smoke_B"]
NUM_ACTIONS = len(ACTIONS)
FEATURE_DIM = 23


# ── Dataset ────────────────────────────────────────────────────────────────

class RoundSequenceDataset(Dataset):
    """
    Each sample = one round, stored as:
      X : (T, feature_dim) float32
      y : (T,)             int64    action labels per tick
    """

    def __init__(
        self,
        X: np.ndarray,        # (N, feature_dim)
        y: np.ndarray,        # (N,)
        seq_len: int = 32,    # fixed window length for batching
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_seq = self.y[idx : idx + self.seq_len]
        # Pad if too short
        pad = self.seq_len - len(x_seq)
        if pad > 0:
            x_seq = F.pad(x_seq, (0, 0, 0, pad))
            y_seq = F.pad(y_seq, (0, pad), value=-1)   # -1 = ignore index
        return x_seq, y_seq


# ── Model ──────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CSGODecisionTransformer(nn.Module):
    """
    Transformer encoder that maps a game state sequence to action logits.

    Architecture:
        Input features (23-dim)
            → Linear projection (d_model)
            → Positional Encoding
            → N × Transformer Encoder Layers
            → Linear head (num_actions)
    
    Parameters
    ----------
    feature_dim  : dimension of input feature vector (default 23)
    d_model      : internal transformer dimension
    nhead        : number of attention heads
    num_layers   : number of transformer encoder blocks
    num_actions  : number of output action classes
    dropout      : dropout rate
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_actions: int = NUM_ACTIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim)
        Returns logits : (batch, seq_len, num_actions)
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)

    def predict(self, x: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Convenience method: returns (action_names, probabilities)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)   # last timestep
            pred_idx = probs.argmax(dim=-1).cpu().numpy()
        action_names = [ACTIONS[i] for i in pred_idx]
        return action_names, probs


# ── Trainer ────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains CSGODecisionTransformer with:
      - Standard cross-entropy (supervised mode)
      - Pseudo-label semi-supervised mode (future)
    
    Example
    -------
    >>> trainer = ModelTrainer(config_path="configs/model_config.yaml")
    >>> trainer.train(X_train, y_train)
    >>> trainer.save("checkpoints/best_model.pt")
    """

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSGODecisionTransformer(
            feature_dim=self.cfg.get("feature_dim", FEATURE_DIM),
            d_model=self.cfg.get("d_model", 128),
            nhead=self.cfg.get("nhead", 4),
            num_layers=self.cfg.get("num_layers", 3),
            num_actions=NUM_ACTIONS,
            dropout=self.cfg.get("dropout", 0.1),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.get("lr", 1e-3),
            weight_decay=self.cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.get("epochs", 50)
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        dataset = RoundSequenceDataset(X, y, seq_len=self.cfg.get("seq_len", 32))
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.get("batch_size", 64),
            shuffle=True,
            num_workers=0,
        )

        losses = []
        epochs = self.cfg.get("epochs", 50)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)  # (B, T, num_actions)

                # Flatten for cross-entropy; ignore -1 padding
                B, T, C = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, C),
                    y_batch.view(B * T),
                    ignore_index=-1,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

        return losses

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded ← {path}")
