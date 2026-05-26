# Website Category Classification

Single-label classification of websites into 25 categories from their page
content. Given a URL, the system fetches the page, extracts the main text from its
HTML, and predicts the category. Three model families are compared: a TF-IDF
linear baseline, a from-scratch CNN/BiLSTM, and a fine-tuned ModernBERT encoder.

**Ege Seçgin · Elia Salerno** — Deep Learning Project Work (ZHAW)

The full report (task, data analysis, approach, experiments, results, and
discussion) is in **`dl_report.pdf`**. This file is a short usage guide.

## Setup

Requires Python 3.13 and the [`uv`](https://docs.astral.sh/uv/) package manager.

```bash
uv sync
```

Get the dataset from Hugging Face [massimilianowosz/website_categories](https://huggingface.co/datasets/massimilianowosz/website_categories):

```bash
uv run python -m wcc.download
```

## Running

```bash
# unit tests
uv run pytest

# run notebooks with jupyterlab or the VSCode extensions, run in order 01 → 05
uv run jupyter lab

# build the dataset
uv run python -m wcc.data

# classify a live URL
uv run wcc-fetch https://www.ventanasystems.com/software/vensim/ --top-k 5    
```

The cleaned train / validation / test splits are included in
`artifacts/processed/`, so the notebooks run without re-downloading or
re-extracting the raw dataset. To rebuild the dataset from raw HTML:
`uv run python -m wcc.data`. Live classification uses the fine-tuned ModernBERT
checkpoint in `artifacts/modernbert/`.

## Structure

```
src/wcc/                   core package: HTML extraction, data pipeline, models, training, inference
notebooks/                 01_eda, 02_baseline_tfidf, 03_scratch_model, 04_modernbert_finetune, 05_results_ablations
tests/                     unit tests (extraction, cleaning, models, metrics, inference)
figures/                   plots used in the report
artifacts/                 processed splits, result metrics, trained model
dl_report.pdf              project report
pyproject.toml, uv.lock    pinned environment
```
