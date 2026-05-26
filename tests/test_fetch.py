"""Unit tests for the inference pipeline's offline behaviour."""

import pytest

from wcc.fetch import CategoryClassifier, FetchError, fetch_html


def test_fetch_html_raises_on_unresolvable_host():
    with pytest.raises(FetchError):
        fetch_html("https://this-domain-does-not-exist.invalid", timeout=5.0)


def test_classifier_raises_clear_error_when_no_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one first"):
        CategoryClassifier(tmp_path / "missing-model")
