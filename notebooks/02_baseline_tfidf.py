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
# # 02: Classical Baseline, TF-IDF + Linear Models
#
# A non-deep reference point. Nandanwar & Choudhary (2023) report classical and
# recurrent baselines below their fine-tuned BERT, which supports the expected
# ordering from classical methods to transformers. This notebook establishes the
# classical rung so later notebooks can quantify the transformer's added value.

# %%
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from wcc.data import PROCESSED_DIR, load_processed
from wcc.metrics import compute_metrics, confusion, plot_confusion_matrix, print_report

train, val, test, label_map = load_processed()
label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
RESULTS_DIR = PROCESSED_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROCESSED_DIR.parent.parent / "figures"  # tracked, repo root
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## TF-IDF features
#
# Word 1-2 grams, English stop-words removed, sublinear term frequency.

# %%
vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), stop_words="english",
                      min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(train["content"])
Xva = vec.transform(val["content"])
Xte = vec.transform(test["content"])
ytr, yva, yte = train["label"], val["label"], test["label"]
print("TF-IDF matrix:", Xtr.shape)

# %% [markdown]
# ## Model selection on the validation split
#
# `class_weight='balanced'` is the linear-model analogue of our weighted loss.

# %%
candidates = {
    "LogReg":    LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
    "LinearSVC": LinearSVC(C=1.0, class_weight="balanced"),
}
val_scores = {}
for name, clf in candidates.items():
    clf.fit(Xtr, ytr)
    m = compute_metrics(yva, clf.predict(Xva), label_names)
    val_scores[name] = m["macro_f1"]
    print(f"{name:<10} val macro-F1 = {m['macro_f1']:.4f}  acc = {m['accuracy']:.4f}")

best_name = max(val_scores, key=val_scores.get)
print(f"\nselected: {best_name}")

# %% [markdown]
# ## Final evaluation on the held-out test split
#
# The test set is touched exactly once.

# %%
best = candidates[best_name]
test_metrics = compute_metrics(yte, best.predict(Xte), label_names)
print_report(test_metrics, f"TF-IDF + {best_name}: TEST")

result = {"model": f"tfidf_{best_name.lower()}", "val_macro_f1": val_scores[best_name],
          "test_macro_f1": test_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"]}
(RESULTS_DIR / "baseline.json").write_text(json.dumps(result, indent=2))
result

# %%
cm = confusion(yte, best.predict(Xte), len(label_names))
ax = plot_confusion_matrix(cm, label_names, f"TF-IDF + {best_name}: test confusion")
ax.figure.tight_layout()
ax.figure.savefig(FIG_DIR / "confusion_baseline.png", dpi=150)

# %% [markdown]
# ## Takeaway
#
# A tuned TF-IDF and linear model is a competitive baseline on this corpus. The
# from-scratch DL model (notebook 03) may not beat it, and a result where it does
# not is a valid finding given the corpus size.
