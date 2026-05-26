"""
Evaluation metrics and result visualization.
The headline metric is macro-F1, because the dataset is imbalanced (Health 1,093
versus Adult 131). It is reported alongside accuracy and a per-class confusion
matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> dict:
    """Compute accuracy, macro/weighted F1 and per-class precision/recall/F1.

    ``labels`` is the ordered list of class names (index == class id).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    idx = list(range(len(labels)))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average=None, zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in idx
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
    }


def confusion(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> np.ndarray:
    """Return the confusion matrix (rows = true, cols = predicted)."""
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def print_report(metrics: dict, title: str = "Results") -> None:
    """Pretty-print a metrics dict to stdout, classes sorted by F1 ascending."""
    print(f"\n=== {title} ===")
    print(f"  accuracy    : {metrics['accuracy']:.4f}")
    print(f"  macro-F1    : {metrics['macro_f1']:.4f}")
    print(f"  weighted-F1 : {metrics['weighted_f1']:.4f}")
    print(f"  {'class':<26}{'P':>7}{'R':>8}{'F1':>8}{'n':>7}")
    items = sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["f1"])
    for name, m in items:
        print(
            f"  {name:<26}{m['precision']:>7.3f}{m['recall']:>8.3f}"
            f"{m['f1']:>8.3f}{m['support']:>7d}"
        )


def plot_confusion_matrix(
    cm: np.ndarray, labels: Sequence[str], title: str = "Confusion matrix", ax=None
):
    """Plot a row-normalized confusion matrix heatmap. Returns the matplotlib Axes.

    Each row (true class) sums to 1, so a dark diagonal indicates high per-class
    recall. ``Blues`` keeps the diagonal dark on white for readability at 25
    classes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        cm_norm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "row-normalized fraction", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.tick_params(labelsize=8)
    return ax


def plot_training_curves(history: Sequence[dict], title: str, save_path=None):
    """Plot loss and validation macro-F1 against epoch. Returns the Figure.

    ``history`` is the per-epoch list of dicts from the training loops. The best
    epoch (peak val macro-F1, the early-stopping checkpoint) is marked.
    """
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    f1 = [h["val_macro_f1"] for h in history]
    best_i = int(np.argmax(f1))

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    if all("val_loss" in h for h in history):
        ax_loss.plot(epochs, [h["val_loss"] for h in history], "s-", label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_loss.legend()

    ax_f1.plot(epochs, f1, "o-", color="seagreen")
    ax_f1.scatter([epochs[best_i]], [f1[best_i]], s=140, color="crimson", zorder=5,
                  label=f"best (epoch {epochs[best_i]}, {f1[best_i]:.3f})")
    ax_f1.set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1")
    ax_f1.legend()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_per_class_f1(per_class: dict, title: str, save_path=None):
    """Horizontal bar chart of per-class F1, sorted ascending. Returns the Figure.

    ``per_class`` is the ``per_class`` block from :func:`compute_metrics`. Bars
    are annotated with class support so small classes are visible.
    """
    import matplotlib.pyplot as plt

    items = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
    names = [k for k, _ in items]
    f1 = [v["f1"] for _, v in items]
    support = [v["support"] for _, v in items]
    macro = float(np.mean(f1))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(names, f1, color="steelblue")
    ax.axvline(macro, color="crimson", ls="--", lw=1.5, label=f"macro-F1 = {macro:.3f}")
    for i, (val, n) in enumerate(zip(f1, support)):
        ax.text(val + 0.01, i, f"n={n}", va="center", fontsize=8, color="dimgray")
    ax.set(xlabel="F1 score", title=title, xlim=(0, 1.05))
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ablations(ablations: dict, save_path=None):
    """Grid of bar charts, one panel per ablation axis. Returns the Figure.

    ``ablations`` maps an axis name (e.g. ``"learning_rate"``) to a dict mapping
    each setting to its validation macro-F1. The best setting per axis is
    highlighted.
    """
    import matplotlib.pyplot as plt

    axes_data = list(ablations.items())
    n = len(axes_data)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axs = np.asarray(axs).reshape(-1)

    for ax, (axis_name, res) in zip(axs, axes_data):
        settings = list(res.keys())
        scores = [res[s] for s in settings]
        best = int(np.argmax(scores))
        colors = ["crimson" if i == best else "steelblue" for i in range(len(scores))]
        ax.bar(range(len(settings)), scores, color=colors)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(settings, rotation=20, ha="right", fontsize=8)
        ax.set_title(axis_name.replace("_", " "))
        ax.set_ylabel("val macro-F1")
        for i, s in enumerate(scores):
            ax.text(i, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(scores) * 1.18)

    for ax in axs[n:]:
        ax.set_visible(False)
    fig.suptitle("ModernBERT ablations: validation macro-F1 (best setting in red)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
