"""Unit tests for HTML -> content extraction and garbage detection."""

from wcc.extract import (
    MIN_CONTENT_CHARS,
    extract_content,
    is_garbage_html,
    normalize_whitespace,
)

_ARTICLE = " ".join(
    [
        "The red panda is a small mammal native to the eastern Himalayas.",
        "It has dense reddish-brown fur and a ringed tail used for balance.",
        "Red pandas are largely herbivorous and feed mainly on bamboo leaves.",
        "They are solitary animals and are most active at dawn and dusk.",
    ]
    * 4
)

GOOD_HTML = f"""<!DOCTYPE html><html lang="en"><head><title>Red Panda</title>
<style>.x{{color:red}}</style></head><body>
<nav>Home About Contact</nav>
<article><h1>The Red Panda</h1><p>{_ARTICLE}</p></article>
<footer>Copyright 2024. All rights reserved.</footer>
<script>console.log('tracking');</script></body></html>"""

CLOUDFLARE_HTML = """<!DOCTYPE html><html><head><title>Just a moment...</title></head>
<body>Checking your browser before accessing the site. cf-browser-verification
Please enable JavaScript and cookies to continue. Ray ID: abc123</body></html>"""

NOTFOUND_HTML = (
    "<html><head><title>404 Not Found</title></head>"
    "<body><h1>404 Not Found</h1><p>Page not found.</p></body></html>"
)

PARKED_HTML = (
    "<html><body><h1>This domain is for sale</h1>"
    "<p>Buy this domain today.</p></body></html>" + "x" * 300
)


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("  a\n\t b   c ") == "a b c"


def test_extract_content_recovers_article_text():
    content = extract_content(GOOD_HTML)
    assert content is not None
    assert "red panda" in content.lower()
    assert len(content) >= MIN_CONTENT_CHARS


def test_extract_content_drops_scripts_and_styles():
    content = extract_content(GOOD_HTML)
    assert content is not None
    assert "console.log" not in content
    assert "color:red" not in content


def test_garbage_detection_flags_cloudflare():
    assert is_garbage_html(CLOUDFLARE_HTML)
    assert extract_content(CLOUDFLARE_HTML) is None


def test_garbage_detection_flags_error_pages():
    assert is_garbage_html(NOTFOUND_HTML)
    assert extract_content(NOTFOUND_HTML) is None


def test_garbage_detection_flags_parked_domains():
    assert is_garbage_html(PARKED_HTML)


def test_garbage_detection_flags_empty_and_tiny_html():
    assert is_garbage_html(None)
    assert is_garbage_html("")
    assert is_garbage_html("<html></html>")


def test_real_article_is_not_garbage():
    assert not is_garbage_html(GOOD_HTML)


def test_short_page_yields_none():
    tiny = "<html><body><p>Hello world.</p></body></html>"
    assert extract_content(tiny) is None
