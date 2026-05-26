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
# # 03: From-Scratch Deep Model (CNN / BiLSTM)
#
# A PyTorch text classifier trained from random initialization: an embedding layer,
# a multi-kernel 1-D CNN (or BiLSTM with attention), and an MLP head. This
# satisfies the "build/adapt a model" requirement. We run light hyperparameter
# exploration and reserve the compute budget for the ModernBERT ablations.

# %%
import json
from torch.utils.data import DataLoader
import torch.nn as nn

from wcc.data import PROCESSED_DIR, load_processed
from wcc.datasets import ScratchTextDataset, scratch_collate
from wcc.metrics import (
    compute_metrics,
    confusion,
    plot_confusion_matrix,
    plot_training_curves,
    print_report,
)
from wcc.train import _evaluate_scratch, get_device, train_scratch_model

train, val, test, label_map = load_processed()
label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
RESULTS_DIR = PROCESSED_DIR.parent / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROCESSED_DIR.parent.parent / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

tr_args = (train["content"].tolist(), train["label"].tolist(),
           val["content"].tolist(), val["label"].tolist(), label_names)

# %% [markdown]
# ## Architecture comparison: CNN vs BiLSTM
#
# Both trained with weighted cross-entropy and early stopping on val macro-F1.

# %%
runs = {}
for arch in ("cnn", "bilstm"):
    model, tok, hist = train_scratch_model(*tr_args, arch=arch, epochs=12, batch_size=64)
    runs[arch] = {"model": model, "tok": tok, "history": hist,
                  "best_val_f1": max(h["val_macro_f1"] for h in hist)}
    print(f"{arch}: best val macro-F1 = {runs[arch]['best_val_f1']:.4f}\n")

# %% [markdown]
# ## Training curves
#
# Loss and validation macro-F1 per epoch for each architecture. The
# early-stopping checkpoint (peak val macro-F1) is marked.

# %%
for arch in ("cnn", "bilstm"):
    plot_training_curves(
        runs[arch]["history"],
        f"From-scratch ({arch}): training curves",
        save_path=FIG_DIR / f"training_curves_scratch_{arch}.png",
    )

# %% [markdown]
# ## Hyperparameter exploration
#
# Vary embedding dimension and dropout on the better architecture. The
# arch-comparison run (default kwargs) is the first candidate, and the grid adds
# three more. The best candidate by validation macro-F1 is carried forward to the
# test set, so the tuning selects the final model.

# %%
best_arch = max(runs, key=lambda a: runs[a]["best_val_f1"])
print(f"continuing with: {best_arch}")
# Three configurations, each distinct from the arch-comparison default
# (embed_dim=200, dropout=0.5), so the grid evaluates four unique settings.
grid = [
    {"embed_dim": 128, "dropout": 0.5},
    {"embed_dim": 128, "dropout": 0.3},
    {"embed_dim": 200, "dropout": 0.3},
]
candidates = [{"kwargs": {}, "model": runs[best_arch]["model"],
               "tok": runs[best_arch]["tok"], "history": runs[best_arch]["history"],
               "best_val_f1": runs[best_arch]["best_val_f1"]}]
for kw in grid:
    model, tok, hist = train_scratch_model(*tr_args, arch=best_arch, epochs=12,
                                           batch_size=64, model_kwargs=kw, verbose=False)
    val_f1 = max(h["val_macro_f1"] for h in hist)
    candidates.append({"kwargs": kw, "model": model, "tok": tok,
                       "history": hist, "best_val_f1": val_f1})
    print(f"{kw}  ->  val macro-F1 = {val_f1:.4f}")

best = max(candidates, key=lambda c: c["best_val_f1"])
print(f"\nselected: arch={best_arch}, kwargs={best['kwargs'] or 'defaults'}  "
      f"(val macro-F1 = {best['best_val_f1']:.4f})")

# %% [markdown]
# ## Test-set evaluation of the best from-scratch model

# %%
test_loader = DataLoader(
    ScratchTextDataset(test["content"].tolist(), test["label"].tolist(), best["tok"], 400),
    batch_size=64, collate_fn=scratch_collate,
)
yt, yp, _ = _evaluate_scratch(best["model"], test_loader, get_device(), nn.CrossEntropyLoss())
test_metrics = compute_metrics(yt, yp, label_names)
print_report(test_metrics, f"From-scratch ({best_arch}): TEST")

(RESULTS_DIR / "scratch.json").write_text(json.dumps(
    {"model": f"scratch_{best_arch}", "kwargs": best["kwargs"],
     "test_macro_f1": test_metrics["macro_f1"],
     "test_accuracy": test_metrics["accuracy"],
     "history": best["history"]}, indent=2))

# %%
cm = confusion(yt, yp, len(label_names))
ax = plot_confusion_matrix(cm, label_names, f"From-scratch ({best_arch}): test confusion")
ax.figure.tight_layout()
ax.figure.savefig(FIG_DIR / "confusion_scratch.png", dpi=150)

# %% [markdown]
# ## Takeaway
#
# On this corpus a from-scratch CNN/BiLSTM learns a usable signal but may not beat
# the TF-IDF baseline, because limited data bounds a randomly-initialized
# embedding. The gain from large-scale pre-training is what notebook 04 tests.
