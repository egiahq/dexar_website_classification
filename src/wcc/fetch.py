"""Inference pipeline: URL to fetch to extract to ModernBERT to top-k categories.

Run as:

    uv run python -m wcc.fetch https://example.com
    uv run wcc-fetch https://example.com --top-k 3

Live fetching is best-effort. Timeouts, JavaScript-heavy single-page apps, and
bot walls are caught and surfaced as a clear error rather than a wrong prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import torch

from wcc.extract import extract_content
from wcc.train import ARTIFACTS

DEFAULT_MODEL_DIR = ARTIFACTS / "modernbert"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched or yields no usable content."""


def fetch_html(url: str, timeout: float = 20.0) -> str:
    """Download a page's HTML, following redirects. Raises :class:`FetchError`."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc
    return resp.text


class CategoryClassifier:
    """Loads a fine-tuned ModernBERT checkpoint and classifies page content."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, max_length: int = 1024):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No model at {model_dir}. Train one first: "
                "uv run python -m wcc.train modernbert"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir)
            .to(self.device)
            .eval()
        )
        self.max_length = max_length
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict_text(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the ``top_k`` (category, probability) pairs for raw text."""
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        k = min(top_k, probs.numel())
        top = torch.topk(probs, k)
        return [
            (self.id2label[int(i)], float(p))
            for p, i in zip(top.values, top.indices)
        ]

    def classify_url(self, url: str, top_k: int = 5) -> dict:
        """Fetch ``url``, extract its content and classify it.

        Uses a lenient extraction threshold, so unlike the training pipeline a
        thin but real page still yields a best-effort prediction. Only bot walls,
        error pages, and JavaScript-only shells are rejected.
        """
        html = fetch_html(url)
        content = extract_content(html, min_chars=30)
        if content is None:
            raise FetchError(
                f"no usable content extracted from {url} "
                "(bot wall, error page, or JS-only site)"
            )
        return {
            "url": url,
            "content_chars": len(content),
            "predictions": self.predict_text(content, top_k),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a website by URL.")
    parser.add_argument("url", help="website URL to fetch and classify")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    try:
        clf = CategoryClassifier(args.model_dir)
        result = clf.classify_url(args.url, top_k=args.top_k)
    except (FetchError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nURL: {result['url']}  ({result['content_chars']} chars extracted)")
    print(f"Top-{args.top_k} categories:")
    for rank, (cat, prob) in enumerate(result["predictions"], 1):
        print(f"  {rank}. {cat:<26} {prob:6.2%}")


if __name__ == "__main__":
    main()
