"""Data preparation: raw parquet shards -> cleaned, de-leaked, split datasets.

Pipeline (run with ``uv run python -m wcc.data``):

1. Concatenate the 5 train + 2 test parquet shards.
2. Extract main content from raw HTML (:mod:`wcc.extract`).
3. Drop garbage pages and extractions shorter than ``MIN_CONTENT_CHARS``.
4. Deduplicate by normalized URL and by content hash.
5. Keep English-only pages (langdetect); report the non-English share.
6. De-leak: drop from train any URL that also appears in test.
7. Carve a stratified 90/10 train/val split from cleaned train. The official
   test split is the held-out set. It passes through the same cleaning filters
   as train, but is never used for tuning or model selection.
8. Persist splits, label map, split indices and a cleaning report to
   ``artifacts/processed/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.model_selection import train_test_split

from wcc.extract import extract_content, is_garbage_html


DetectorFactory.seed = 0

SEED = 42
VAL_FRACTION = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "massimilianowosz_website_categories"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"

_WWW_RE = re.compile(r"^www\.")
_SCHEME_RE = re.compile(r"^[a-z]+://")





def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication and leakage checks.

    Lower-cases, strips the scheme, a leading ``www.`` and any trailing slash.
    """
    u = str(url).strip().lower()
    u = _SCHEME_RE.sub("", u)
    u = _WWW_RE.sub("", u)
    return u.rstrip("/")


def content_hash(text: str) -> str:
    """Stable hash of whitespace/case-normalized content, for exact-duplicate removal."""
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """Detect the language of ``text`` (first 2000 chars); ``"unknown"`` on failure."""
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"





def load_raw(split: str) -> pd.DataFrame:
    """Concatenate the raw parquet shards for ``split`` ('train' or 'test')."""
    shards = sorted(RAW_DIR.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split} parquet shards under {RAW_DIR}")
    df = pd.concat(
        (pd.read_parquet(f, columns=["url", "text", "main_category"]) for f in shards),
        ignore_index=True,
    )
    return df.rename(columns={"text": "html", "main_category": "category"})





def _extract_one(html: str | None) -> str | None:
    return extract_content(html)


def _extract_split(split: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extract content for every row, with an on-disk cache per split."""
    cache_path = CACHE_DIR / f"extracted_{split}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(df) and "garbage" in cached.columns:
            print(f"  [{split}] using cached extractions ({len(cached)} rows)")
            return cached

    print(f"  [{split}] extracting content from {len(df)} pages ...")
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        contents = list(pool.map(_extract_one, df["html"].tolist(), chunksize=32))
    out = pd.DataFrame(
        {
            "url": df["url"].astype(str),
            "category": df["category"].astype(str),
            "html_len": df["html"].str.len().fillna(0).astype(int),


            "garbage": df["html"].map(is_garbage_html),
            "content": contents,
        }
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"  [{split}] extraction done in {time.time() - t0:.0f}s")
    return out





def build_dataset() -> dict:
    """Run the full cleaning pipeline and persist the result. Returns the report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": SEED, "val_fraction": VAL_FRACTION}

    raw_train, raw_test = load_raw("train"), load_raw("test")
    report["raw_train_rows"] = len(raw_train)
    report["raw_test_rows"] = len(raw_test)

    train = _extract_split("train", raw_train)
    test = _extract_split("test", raw_test)

    def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
        n0 = len(df)
        stats: dict = {"initial": n0}

        # extract_content returned None either because the raw HTML was a garbage
        # page or because the extracted text fell below MIN_CONTENT_CHARS. The
        # `garbage` flag separates the two so the report records each cause.
        dropped = df["content"].isna()
        stats["dropped_garbage"] = int((dropped & df["garbage"]).sum())
        stats["dropped_short"] = int((dropped & ~df["garbage"]).sum())
        df = df[~dropped].copy()


        df["norm_url"] = df["url"].map(normalize_url)
        before = len(df)
        df = df.drop_duplicates(subset="norm_url", keep="first")
        stats["dropped_dup_url"] = before - len(df)



        df["content_hash"] = df["content"].map(content_hash)
        before = len(df)
        df = df.drop_duplicates(subset="content_hash", keep="first")
        stats["dropped_dup_content"] = before - len(df)


        df["lang"] = df["content"].map(detect_language)
        non_en = int((df["lang"] != "en").sum())
        stats["non_english"] = non_en
        stats["non_english_share"] = round(non_en / max(len(df), 1), 4)
        df = df[df["lang"] == "en"].copy()

        stats["kept"] = len(df)
        report[f"clean_{name}"] = stats
        print(f"  [{name}] {n0} -> {len(df)} rows  {stats}")
        return df

    print("Cleaning train ...")
    train = clean(train, "train")
    print("Cleaning test ...")
    test = clean(test, "test")


    test_urls = set(test["norm_url"])
    before = len(train)
    train = train[~train["norm_url"].isin(test_urls)].copy()
    report["train_test_url_overlap_removed"] = before - len(train)


    leakage = len(set(train["norm_url"]) & set(test["norm_url"]))
    report["leakage"] = leakage
    assert leakage == 0, f"Train/test URL leakage detected: {leakage}"


    categories = sorted(set(train["category"]) | set(test["category"]))
    label_map = {c: i for i, c in enumerate(categories)}
    report["num_classes"] = len(categories)
    for df in (train, test):
        df["label"] = df["category"].map(label_map).astype(int)


    tr_idx, va_idx = train_test_split(
        train.index.to_numpy(),
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=train["label"].to_numpy(),
    )
    train_split = train.loc[tr_idx].reset_index(drop=True)
    val_split = train.loc[va_idx].reset_index(drop=True)
    test_split = test.reset_index(drop=True)

    report["split_sizes"] = {
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
    }

    cols = ["url", "norm_url", "content", "category", "label", "html_len"]
    train_split[cols].to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_split[cols].to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_split[cols].to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    (PROCESSED_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (PROCESSED_DIR / "split_indices.json").write_text(
        json.dumps({"train": tr_idx.tolist(), "val": va_idx.tolist()}, indent=2)
    )
    (PROCESSED_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2))

    return report





def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the persisted train/val/test splits and the label map.

    Raises a clear error if :func:`build_dataset` has not been run yet.
    """
    if not (PROCESSED_DIR / "train.parquet").exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: uv run python -m wcc.data"
        )
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    label_map = json.loads((PROCESSED_DIR / "label_map.json").read_text())
    return train, val, test, label_map


def main() -> None:
    report = build_dataset()
    print("\n=== Dataset build report ===")
    print(json.dumps(report, indent=2))
    print(f"\nleakage == {report['leakage']}  (must be 0)")
    print("Splits:", report["split_sizes"])


if __name__ == "__main__":
    main()
