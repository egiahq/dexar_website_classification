"""Unit tests for the from-scratch CNN / BiLSTM text classifiers."""

import pytest
import torch

from wcc.datasets import PAD_ID
from wcc.models import (
    BiLSTMTextClassifier,
    CNNTextClassifier,
    build_scratch_model,
    count_parameters,
)

VOCAB, CLASSES = 50, 7


def test_cnn_forward_shape():
    model = CNNTextClassifier(VOCAB, CLASSES)
    ids = torch.randint(0, VOCAB, (4, 20))
    assert model(ids).shape == (4, CLASSES)


def test_cnn_handles_sequence_shorter_than_kernel():
    # Kernels default to (3, 4, 5). A length-1 sequence (the dataset's empty-text
    # fallback) must still produce a valid logit vector.
    model = CNNTextClassifier(VOCAB, CLASSES)
    ids = torch.randint(0, VOCAB, (2, 1))
    assert model(ids).shape == (2, CLASSES)


def test_bilstm_forward_shape():
    model = BiLSTMTextClassifier(VOCAB, CLASSES)
    ids = torch.randint(1, VOCAB, (4, 20))
    lengths = torch.tensor([20, 15, 8, 3])
    assert model(ids, lengths).shape == (4, CLASSES)


def test_bilstm_attention_ignores_padding_without_nan():
    model = BiLSTMTextClassifier(VOCAB, CLASSES)
    ids = torch.randint(1, VOCAB, (3, 12))
    ids[0, 4:] = PAD_ID                       
    ids[1, 9:] = PAD_ID
    lengths = torch.tensor([4, 9, 12])
    out = model(ids, lengths)
    assert out.shape == (3, CLASSES)
    assert torch.isfinite(out).all()


def test_build_scratch_model_factory():
    assert isinstance(build_scratch_model("cnn", VOCAB, CLASSES), CNNTextClassifier)
    assert isinstance(build_scratch_model("bilstm", VOCAB, CLASSES), BiLSTMTextClassifier)
    assert isinstance(build_scratch_model("lstm", VOCAB, CLASSES), BiLSTMTextClassifier)


def test_build_scratch_model_rejects_unknown_arch():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_scratch_model("transformer", VOCAB, CLASSES)


def test_count_parameters_positive():
    assert count_parameters(CNNTextClassifier(VOCAB, CLASSES)) > 0
