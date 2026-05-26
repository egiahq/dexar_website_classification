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
# # 04: Main Model, Fine-Tuning ModernBERT
#
# `answerdotai/ModernBERT-base` (2024 encoder, native long context) with a
# classification head. AdamW, linear warmup and decay, inverse-frequency weighted
# cross-entropy for imbalance, bf16 mixed precision, gradient accumulation for
# the 12 GB RTX 3080 Ti, early stopping on validation macro-F1.
#
# > GPU training. A full run takes tens of minutes per configuration.

# %%
import json
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import DataCollatorWithPadding

from wcc.data import PROCESSED_DIR, load_processed
from wcc.datasets import TransformerTextDataset
from wcc.metrics import (
    compute_metrics,
    confusion,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_training_curves,
    print_report,
)
from wcc.train import ARTIFACTS, _evaluate_transformer, get_device, train_transformer

train, val, test, label_map = load_processed()
label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
RESULTS_DIR = ARTIFACTS / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ARTIFACTS.parent / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

tr_args = (train["content"].tolist(), train["label"].tolist(),
           val["content"].tolist(), val["label"].tolist(), label_names)

# %% [markdown]
# ## Fine-tune
#
# `freeze='full'` fine-tunes the entire encoder plus the classification head,
# the highest-scoring setup and within the 12 GB budget at this batch size. The
# frozen and last-N alternatives are compared in notebook 05.

# %%
import statistics

FINAL_CONFIG = dict(
    max_length=1024,         # nb05 ablation winner: sequence_length
    lr=5e-5,                 # nb05 ablation winner: learning_rate
    pooling="cls",           # nb05 ablation winner: pooling
    use_class_weights=True,  # nb05 ablation winner: class_weighting
    freeze="full",           # nb05 ablation winner: fine-tuning depth
)
SEEDS = (42, 43, 44)
TRAIN_KW = dict(epochs=5, batch_size=8, grad_accum=2, **FINAL_CONFIG)

seed_runs = []
for s in SEEDS:
    model, tokenizer, hist = train_transformer(
        *tr_args,
        seed=s,
        verbose=(s == SEEDS[0]),
        save_dir=ARTIFACTS / "modernbert" if s == SEEDS[0] else None,
        **TRAIN_KW,
    )
    test_loader = DataLoader(
        TransformerTextDataset(test["content"].tolist(), test["label"].tolist(),
                               tokenizer, FINAL_CONFIG["max_length"]),
        batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer),
    )
    yt_s, yp_s, _ = _evaluate_transformer(model, test_loader, get_device(),
                                          nn.CrossEntropyLoss())
    m_s = compute_metrics(yt_s, yp_s, label_names)
    seed_runs.append({"seed": s, "metrics": m_s, "history": hist,
                      "y_true": yt_s, "y_pred": yp_s})
    print(f"seed {s}: test macro-F1 = {m_s['macro_f1']:.4f}  "
          f"acc = {m_s['accuracy']:.4f}")

# Seed 42 is the reference run (saved checkpoint, plotted curves and confusion).
ref = seed_runs[0]
history = ref["history"]
for h in history:
    print(h)

# %% [markdown]
# ## Training curves
#
# Train/validation loss and validation macro-F1 per epoch. Validation macro-F1
# peaks early and then declines as the model overfits, so early stopping keeps the
# best checkpoint (marked in red).

# %%
plot_training_curves(
    history,
    "ModernBERT (full fine-tune): training curves",
    save_path=FIG_DIR / "training_curves_modernbert.png",
)

# %% [markdown]
# ## Held-out test evaluation: mean +/- std over three seeds
#
# The headline number is the mean test macro-F1 across the three seeds. The
# per-class table and confusion matrix below are from the seed-42 reference run.

# %%
yt, yp = ref["y_true"], ref["y_pred"]
test_metrics = compute_metrics(yt, yp, label_names)
print_report(test_metrics, "ModernBERT (seed 42 reference): TEST")

f1s = [r["metrics"]["macro_f1"] for r in seed_runs]
accs = [r["metrics"]["accuracy"] for r in seed_runs]
mean_f1, std_f1 = statistics.mean(f1s), statistics.pstdev(f1s)
mean_acc, std_acc = statistics.mean(accs), statistics.pstdev(accs)
print(f"3-seed test macro-F1 = {mean_f1:.4f} +/- {std_f1:.4f}   "
      f"accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

(RESULTS_DIR / "modernbert.json").write_text(json.dumps(
    {"model": "modernbert",
     "config": FINAL_CONFIG,
     "seeds": list(SEEDS),
     "test_macro_f1_seeds": f1s,
     "test_macro_f1": mean_f1,
     "test_macro_f1_std": std_f1,
     "test_accuracy": mean_acc,
     "test_accuracy_std": std_acc,
     "per_class": test_metrics["per_class"]}, indent=2))

# %%
cm = confusion(yt, yp, len(label_names))
ax = plot_confusion_matrix(cm, label_names, "ModernBERT: test confusion")
ax.figure.tight_layout()
ax.figure.savefig(FIG_DIR / "confusion_modernbert.png", dpi=150)

# %% [markdown]
# ## Per-class F1
#
# Where the model is strong and where it is weak. Small classes (low `n`) and
# semantically broad categories sit below the macro-F1 line.

# %%
plot_per_class_f1(
    test_metrics["per_class"],
    "ModernBERT: per-class test F1",
    save_path=FIG_DIR / "per_class_f1.png",
)

# %% [markdown]
# ## Takeaway
#
# Large-scale pre-training transfers, so ModernBERT is expected to clear both the
# TF-IDF baseline and the from-scratch model. Residual errors concentrate in
# semantically adjacent classes (Business and Career, Reference and Science) and
# are partly bounded by single-source label noise. See notebook 05.
