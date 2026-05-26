"""Unit tests for training utilities."""

import torch

from wcc.train import compute_class_weights, set_seed


def test_compute_class_weights_mean_is_one():
    weights = compute_class_weights([0, 0, 0, 0, 1, 1], num_classes=2)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_compute_class_weights_upweights_rare_class():
                                                                       
    weights = compute_class_weights([0, 0, 0, 0, 0, 0, 1, 1], num_classes=2)
    assert weights[1] > weights[0]


def test_compute_class_weights_handles_absent_class():
                                                                        
    weights = compute_class_weights([0, 0, 1], num_classes=3)
    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()


def test_set_seed_makes_torch_rng_repeatable():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)
