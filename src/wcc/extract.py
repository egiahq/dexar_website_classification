"""HTML to clean main content extraction.

Uses trafilatura with a BeautifulSoup ``get_text()`` fallback. Residual failure
modes (cookie and copyright notices that read as fluent text) are a known
limitation.
"""

from __future__ import annotations

import re

import trafilatura
from bs4 import BeautifulSoup


MIN_CONTENT_CHARS = 200

_GARBAGE_PATTERNS: tuple[str, ...] = (
    "just a moment...",
    "checking your browser before accessing",
    "cf-browser-verification",
    "attention required! | cloudflare",
    "enable javascript and cookies to continue",
    "ddos protection by cloudflare",
    "ray id:",

    "404 not found",
    "403 forbidden",
    "page not found",
    "access denied",
    "this site can&#39;t be reached",
    "service unavailable",

    "this domain is for sale",
    "buy this domain",
    "the domain name you entered is for sale",
    "this domain is parked",
    "domain parking",

    "you need to enable javascript to run this app",
    "please enable javascript to view",
    "your browser does not support javascript",
)

_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return _WS_RE.sub(" ", text).strip()


def is_garbage_html(html: str | None) -> bool:
    """Return ``True`` if the raw HTML looks like a bot wall, error, or parked page.

    A conservative substring match on a lowercased prefix, so it flags obvious
    non-content pages without discarding short but real websites. Empty or
    trivially small HTML also counts as garbage.
    """
    if not html or len(html) < 200:
        return True
    head = html[:20_000].lower()
    return any(pat in head for pat in _GARBAGE_PATTERNS)


def _trafilatura_extract(html: str) -> str | None:
    """Run trafilatura with settings tuned for recall over precision."""
    try:
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            no_fallback=False,
            deduplicate=True,
        )
    except Exception:
        return None


def _bs4_extract(html: str) -> str | None:
    """Fallback extractor: strip scripts/styles and take visible text."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = normalize_whitespace(text)
    return text or None


def extract_content(html: str | None, min_chars: int = MIN_CONTENT_CHARS) -> str | None:
    """Extract clean main textual content from raw page HTML.

    Returns the cleaned text, or ``None`` when the page is garbage or no usable
    content could be recovered (shorter than ``min_chars``). Training uses the
    strict default. Inference (:mod:`wcc.fetch`) passes a smaller ``min_chars``
    so thin but real pages still get a best-effort prediction.
    """
    if is_garbage_html(html):
        return None
    assert html is not None

    text = _trafilatura_extract(html)
    if text is None or len(text) < min_chars:
        fallback = _bs4_extract(html)

        if fallback is not None and (text is None or len(fallback) > len(text)):
            text = fallback

    if text is None:
        return None
    text = normalize_whitespace(text)
    return text if len(text) >= min_chars else None
