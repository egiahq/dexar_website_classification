"""Unit tests for data-cleaning helpers (URL/content normalization)."""

from wcc.data import content_hash, detect_language, normalize_url


def test_normalize_url_strips_scheme_www_and_slash():
    assert normalize_url("https://www.Example.com/") == "example.com"
    assert normalize_url("http://example.com") == "example.com"
    assert normalize_url("example.com/") == "example.com"


def test_normalize_url_collapses_leakage_variants():
    variants = [
        "https://www.site.org/",
        "http://site.org",
        "site.org/",
        "SITE.ORG",
    ]
    assert len({normalize_url(u) for u in variants}) == 1


def test_content_hash_is_whitespace_and_case_insensitive():
    assert content_hash("Hello   World") == content_hash("hello world")
    assert content_hash("a different page") != content_hash("hello world")


def test_detect_language_identifies_english():
    text = (
        "This is a clearly written English paragraph about web page "
        "classification and natural language processing techniques."
    )
    assert detect_language(text) == "en"
