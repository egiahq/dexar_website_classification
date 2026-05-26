# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01: Exploratory Data Analysis
#
# Website Category Classifier, DL Project Work.
#
# We analyse the cleaned dataset produced by `wcc.data`: 25-class single-label
# website classification. The headline metric is macro-F1 because the class
# distribution is imbalanced (Health versus Adult).
#
# > Run `uv run python -m wcc.data` first to build the processed dataset.

# %%
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from wcc.data import PROCESSED_DIR, load_processed

sns.set_theme(style="whitegrid")
FIG_DIR = PROCESSED_DIR.parent.parent / "figures"  # tracked, repo root
FIG_DIR.mkdir(parents=True, exist_ok=True)

train, val, test, label_map = load_processed()
label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
full = pd.concat([train, val], ignore_index=True)
print(f"train={len(train)}  val={len(val)}  test={len(test)}  classes={len(label_names)}")

# %% [markdown]
# ## Cleaning report
#
# Provenance of every dropped row, and the leakage check.

# %%
report = json.loads((PROCESSED_DIR / "dataset_report.json").read_text())
print(json.dumps(report, indent=2))
assert report["leakage"] == 0, "train/test URL leakage must be zero"
print("\nLeakage check passed: 0 shared URLs between train and test.")

# %% [markdown]
# ## Class distribution
#
# The imbalance motivates macro-F1 as the headline metric and inverse-frequency
# weighted cross-entropy during training.

# %%
counts = full["category"].value_counts()
fig, ax = plt.subplots(figsize=(10, 7))
counts.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("number of pages (train + val)")
ax.set_title(f"Class distribution: imbalance ratio {counts.max() / counts.min():.1f}x")
fig.tight_layout()
fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
plt.show()
print(f"largest: {counts.idxmax()} ({counts.max()})   smallest: {counts.idxmin()} ({counts.min()})")

# %% [markdown]
# ## Content length: raw HTML vs extracted text
#
# The `extract.py` step (trafilatura) strips boilerplate, so the extracted text is
# shorter than the raw HTML. Extraction is a pipeline stage, not free
# preprocessing (Web2Text, BoilerNet).

# %%
full = full.assign(content_len=full["content"].str.len())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(full["html_len"].clip(upper=200_000), bins=60, color="indianred")
axes[0].set(title="Raw HTML length (chars)", xlabel="chars", ylabel="pages")
axes[1].hist(full["content_len"].clip(upper=20_000), bins=60, color="seagreen")
axes[1].set(title="Extracted content length (chars)", xlabel="chars")
fig.tight_layout()
fig.savefig(FIG_DIR / "content_length.png", dpi=150)
plt.show()
print(full[["html_len", "content_len"]].describe().round(0))

# %% [markdown]
# ## Token-length distribution under the ModernBERT tokenizer
#
# This informs the sequence-length choice and quantifies the truncation rate at
# 256, 512, and 1024 tokens, the hyperparameter Nandanwar & Choudhary (2023)
# found most impactful.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
sample = full["content"].sample(min(3000, len(full)), random_state=42).tolist()
tok_lens = pd.Series([len(x) for x in tok(sample, truncation=False)["input_ids"]])

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(tok_lens.clip(upper=2048), bins=60, color="slateblue")
for L in (256, 512, 1024):
    ax.axvline(L, color="black", ls="--", lw=1)
    ax.text(L, ax.get_ylim()[1] * 0.9, f" {L}", rotation=90, va="top")
ax.set(title="ModernBERT token counts per page", xlabel="tokens", ylabel="pages")
fig.tight_layout()
fig.savefig(FIG_DIR / "token_lengths.png", dpi=150)
plt.show()

print("truncation rate (fraction of pages longer than the limit):")
for L in (256, 512, 1024):
    print(f"  {L:>5} tokens: {(tok_lens > L).mean():6.1%}   median tokens: {tok_lens.median():.0f}")

# %% [markdown]
# ## Language mix
#
# We keep English-only pages; the non-English share found during cleaning:

# %%
for split in ("train", "test"):
    s = report[f"clean_{split}"]
    print(f"{split}: non-English {s['non_english']} pages ({s['non_english_share']:.1%}), dropped")

# %% [markdown]
# ## Example extractions
#
# Qualitative check of the HTML->content step.

# %%
for _, row in full.sample(3, random_state=1).iterrows():
    print(f"[{row['category']}]  {row['url']}")
    print(f"  {row['content'][:300]}...")
    print()

# %% [markdown]
# ## Top TF-IDF terms per class
#
# The most discriminative vocabulary per category, a check that the text signal
# separates classes, and a preview of confusable pairs.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), stop_words="english",
                      min_df=5, sublinear_tf=True)
X = vec.fit_transform(full["content"])
terms = vec.get_feature_names_out()
for cat in label_names[:8]:
    mask = (full["category"] == cat).to_numpy()
    mean_tfidf = X[mask].mean(axis=0).A1
    top = mean_tfidf.argsort()[::-1][:10]
    print(f"{cat:<26} {', '.join(terms[i] for i in top)}")

# %% [markdown]
# ## Word clouds for selected classes

# %%
from wordcloud import WordCloud

show = ["Health", "Sports", "Finance", "Adult"]
show = [c for c in show if c in set(full["category"])][:4]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, cat in zip(axes.flat, show):
    text = " ".join(full.loc[full["category"] == cat, "content"].head(300))
    wc = WordCloud(width=600, height=400, background_color="white",
                   stopwords=None, collocations=False).generate(text)
    ax.imshow(wc); ax.axis("off"); ax.set_title(cat)
fig.tight_layout()
fig.savefig(FIG_DIR / "wordclouds.png", dpi=150)
plt.show()

# %% [markdown]
# ## Takeaways
#
# The data has 25 classes with class imbalance, which is why macro-F1 is the
# headline metric and the loss is class-weighted. Trafilatura shrinks pages by
# roughly 10 to 20 times, so extraction is a distinct pipeline stage. The
# token-length statistics inform the 256/512/1024 sequence-length ablation. After
# de-leaking there is zero train/test overlap, and the cleaning report is
# reproducible.
