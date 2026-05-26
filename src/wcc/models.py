"""From-scratch PyTorch text classifiers, both trained from random initialization.

:class:`CNNTextClassifier` uses multi-kernel 1-D convolutions with max-over-time
pooling (Kim-2014 style). :class:`BiLSTMTextClassifier` uses a bidirectional LSTM
with additive attention pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wcc.datasets import PAD_ID


class CNNTextClassifier(nn.Module):
    """Embedding, parallel 1-D convolutions, max pool, MLP head."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 200,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor | None = None):
        # ids: (B, T) -> embed -> (B, C=embed_dim, T) for Conv1d.
        x = self.dropout(self.embedding(ids)).transpose(1, 2)
        feats = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.head(self.dropout(torch.cat(feats, dim=1)))


class BiLSTMTextClassifier(nn.Module):
    """Embedding, bidirectional LSTM, additive attention pooling, MLP head."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor):
        mask = ids != PAD_ID  # (B, T)
        x = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # additive attention pooling, masking out padded positions.
        scores = self.attn(out).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask[:, : scores.size(1)], float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (out * weights).sum(dim=1)  # (B, 2H)
        return self.head(self.dropout(pooled))


def build_scratch_model(arch: str, vocab_size: int, num_classes: int, **kwargs):
    """Factory: ``arch`` is ``"cnn"`` or ``"bilstm"``."""
    arch = arch.lower()
    if arch == "cnn":
        return CNNTextClassifier(vocab_size, num_classes, **kwargs)
    if arch in ("bilstm", "lstm"):
        return BiLSTMTextClassifier(vocab_size, num_classes, **kwargs)
    raise ValueError(f"Unknown architecture: {arch!r} (expected 'cnn' or 'bilstm')")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
