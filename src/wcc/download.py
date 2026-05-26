"""Fetch the raw dataset from the HuggingFace Hub into ``data/``.

Run with ``uv run python -m wcc.download``. Downloads the parquet shards of
``massimilianowosz/website_categories`` and places them flat under
``data/massimilianowosz_website_categories/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "massimilianowosz/website_categories"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "massimilianowosz_website_categories"


def download(force: bool = False) -> Path:
    """Download the dataset's parquet shards into :data:`RAW_DIR`.

    Returns the directory. With ``force``, existing shards are overwritten.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["*.parquet"],
    )
    shards = sorted(Path(snapshot).rglob("*.parquet"))
    if not shards:
        raise RuntimeError(f"No parquet shards found in dataset repo {REPO_ID!r}")

    copied = 0
    for src in shards:
        dest = RAW_DIR / src.name
        if force or not dest.exists():
            shutil.copy2(src, dest)
            copied += 1

    print(f"{len(shards)} shards in {RAW_DIR} ({copied} newly written)")
    return RAW_DIR


def main() -> None:
    download()


if __name__ == "__main__":
    main()
