"""Unit tests for evaluation metrics."""

import numpy as np

from wcc.metrics import compute_metrics, confusion

LABELS = ["a", "b", "c"]


def test_compute_metrics_perfect_prediction():
    y = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y, y, LABELS)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    for cls in LABELS:
        assert m["per_class"][cls]["f1"] == 1.0


def test_compute_metrics_per_class_support_sums_to_n():
    y_true = [0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2, 0]
    m = compute_metrics(y_true, y_pred, LABELS)
    assert sum(c["support"] for c in m["per_class"].values()) == len(y_true)
    assert m["per_class"]["a"]["support"] == 2
    assert m["per_class"]["c"]["support"] == 3


def test_compute_metrics_handles_class_with_no_predictions():
                                                                        
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    m = compute_metrics(y_true, y_pred, LABELS)
    assert m["per_class"]["c"]["precision"] == 0.0
    assert m["per_class"]["c"]["recall"] == 0.0
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_confusion_matrix_shape_and_diagonal():
    y = [0, 1, 2, 0, 1, 2]
    cm = confusion(y, y, n_classes=3)
    assert cm.shape == (3, 3)
    assert np.array_equal(np.diag(cm), np.array([2, 2, 2]))
    assert cm.sum() == len(y)
