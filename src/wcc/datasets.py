"""PyTorch datasets and tokenization.

:class:`SimpleTokenizer` with :class:`ScratchTextDataset` provides a word-level
vocabulary for the from-scratch CNN/BiLSTM model. :class:`TransformerTextDataset`
provides HuggingFace sub-word tokenization for ModernBERT fine-tuning.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

_TOKEN_RE = re.compile(r"[a-z0-9]+")

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


def word_tokenize(text: str) -> list[str]:
    """Lower-case word/number tokenizer for the from-scratch model."""
    return _TOKEN_RE.findall(text.lower())


class SimpleTokenizer:
    """Frequency-capped word vocabulary built from the training corpus."""

    def __init__(self, max_vocab: int = 30_000, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.itos: list[str] = [PAD, UNK]
        self.stoi: dict[str, int] = {PAD: PAD_ID, UNK: UNK_ID}

    def fit(self, texts: Sequence[str]) -> "SimpleTokenizer":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(word_tokenize(t))
        for word, freq in counter.most_common():
            if freq < self.min_freq or len(self.itos) >= self.max_vocab:
                break
            self.stoi[word] = len(self.itos)
            self.itos.append(word)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, UNK_ID) for tok in word_tokenize(text)]
        return ids[:max_len]


class ScratchTextDataset(Dataset):
    """Token-id dataset for the from-scratch model.

    ``__getitem__`` returns a (1-D LongTensor of token ids, label) pair. Padding
    to a common length is done by :func:`scratch_collate`.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: SimpleTokenizer,
        max_len: int = 400,
    ):
        self.encoded = [tokenizer.encode(t, max_len) or [UNK_ID] for t in texts]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.encoded[i], dtype=torch.long), self.labels[i]


def scratch_collate(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length id sequences. Returns (ids, lengths, labels)."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    maxlen = int(lengths.max())
    padded = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class TransformerTextDataset(Dataset):
    """Sub-word tokenized dataset for ModernBERT fine-tuning.

    Tokenization is done eagerly in ``__init__`` so epochs are I/O-free.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 1024,
    ):
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }
