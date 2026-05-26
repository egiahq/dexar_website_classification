"""Unit tests for tokenization and the from-scratch torch datasets."""

import torch

from wcc.datasets import (
    PAD_ID,
    UNK_ID,
    ScratchTextDataset,
    SimpleTokenizer,
    scratch_collate,
    word_tokenize,
)


def test_word_tokenize_lowercases_and_splits():
    assert word_tokenize("Hello, WORLD! 123 web-page") == [
        "hello", "world", "123", "web", "page"
    ]


def test_simple_tokenizer_reserves_pad_and_unk():
    tok = SimpleTokenizer().fit(["hello world hello world"])
    assert tok.itos[PAD_ID] == "<pad>"
    assert tok.itos[UNK_ID] == "<unk>"
    assert tok.vocab_size >= 2


def test_simple_tokenizer_min_freq_excludes_rare_words():
    tok = SimpleTokenizer(min_freq=2).fit(["common common rare"])
    assert "common" in tok.stoi
    assert "rare" not in tok.stoi


def test_simple_tokenizer_encode_truncates_and_maps_unknowns():
    tok = SimpleTokenizer(min_freq=1).fit(["alpha beta gamma"])
    assert len(tok.encode("alpha beta gamma delta", max_len=3)) == 3
    assert tok.encode("zzz", max_len=5) == [UNK_ID]


def test_scratch_dataset_returns_long_tensor_and_label():
    tok = SimpleTokenizer(min_freq=1).fit(["alpha beta gamma"])
    ds = ScratchTextDataset(["alpha beta"], [3], tok, max_len=10)
    ids, label = ds[0]
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    assert label == 3


def test_scratch_dataset_empty_text_falls_back_to_unk():
    tok = SimpleTokenizer(min_freq=1).fit(["alpha beta"])
    ds = ScratchTextDataset(["", "!!!"], [0, 1], tok, max_len=10)
    assert ds[0][0].tolist() == [UNK_ID]
    assert ds[1][0].tolist() == [UNK_ID]


def test_scratch_collate_pads_to_batch_max():
    batch = [
        (torch.tensor([5, 6, 7]), 0),
        (torch.tensor([8]), 1),
    ]
    ids, lengths, labels = scratch_collate(batch)
    assert ids.shape == (2, 3)
    assert lengths.tolist() == [3, 1]
    assert labels.tolist() == [0, 1]
    assert ids[1, 1].item() == PAD_ID
    assert ids[1, 2].item() == PAD_ID
