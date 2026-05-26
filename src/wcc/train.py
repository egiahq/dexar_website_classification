"""Training and evaluation loops for both model families.

:func:`train_scratch_model` trains the from-scratch CNN/BiLSTM classifier.
:func:`train_transformer` fine-tunes ``answerdotai/ModernBERT-base``. Both use
inverse-frequency weighted cross-entropy for the class imbalance and early
stopping on validation macro-F1. Run directly to train the main model:

    uv run python -m wcc.train modernbert
    uv run python -m wcc.train scratch --arch cnn
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wcc.data import PROCESSED_DIR, load_processed
from wcc.datasets import (
    ScratchTextDataset,
    SimpleTokenizer,
    TransformerTextDataset,
    scratch_collate,
)
from wcc.metrics import compute_metrics, print_report
from wcc.models import build_scratch_model, count_parameters

MODEL_NAME = "answerdotai/ModernBERT-base"
ARTIFACTS = PROCESSED_DIR.parent





def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs and pin cuDNN to deterministic mode.

    This makes runs repeatable. Bitwise-identical CUDA results would additionally
    require ``torch.use_deterministic_algorithms(True)``, which is not enabled
    because the BiLSTM has no deterministic cuDNN backward kernel and it would
    raise at runtime.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.0."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float)





def _evaluate_scratch(model, loader, device, criterion) -> tuple[list, list, float]:
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    with torch.no_grad():
        for ids, lengths, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids, lengths)
            total_loss += criterion(logits, labels).item() * len(labels)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
    return y_true, y_pred, total_loss / len(loader.dataset)


def train_scratch_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    label_names: list[str],
    *,
    arch: str = "cnn",
    max_len: int = 400,
    max_vocab: int = 30_000,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_class_weights: bool = True,
    patience: int = 3,
    model_kwargs: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
):
    """Train the from-scratch CNN/BiLSTM classifier; returns (model, tokenizer, history)."""
    set_seed(seed)
    device = get_device()
    num_classes = len(label_names)

    tokenizer = SimpleTokenizer(max_vocab=max_vocab).fit(train_texts)
    train_ds = ScratchTextDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = ScratchTextDataset(val_texts, val_labels, tokenizer, max_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=scratch_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=scratch_collate
    )

    model = build_scratch_model(
        arch, tokenizer.vocab_size, num_classes, **(model_kwargs or {})
    ).to(device)
    if verbose:
        print(
            f"{arch} model: {count_parameters(model):,} trainable params, "
            f"vocab={tokenizer.vocab_size}"
        )

    weights = (
        compute_class_weights(train_labels, num_classes).to(device)
        if use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for ids, lengths, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(labels)
        train_loss = running / len(train_ds)

        y_true, y_pred, val_loss = _evaluate_scratch(
            model, val_loader, device, criterion
        )
        m = compute_metrics(y_true, y_pred, label_names)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": m["macro_f1"],
                "val_accuracy": m["accuracy"],
            }
        )
        if verbose:
            print(
                f"epoch {epoch:2d}  train_loss {train_loss:.4f}  "
                f"val_loss {val_loss:.4f}  val_macroF1 {m['macro_f1']:.4f}"
            )

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(
                        f"early stopping at epoch {epoch} (best macro-F1 {best_f1:.4f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, history





def _apply_freezing(model, mode: str, last_n: int = 2) -> None:
    """Freeze parameters according to ``mode``: 'full' | 'head' | 'last_n'."""
    if mode == "full":
        return
    base = model.base_model
    for p in base.parameters():
        p.requires_grad = False
    if mode == "head":
        return
    if mode == "last_n":
        layers = getattr(base, "layers", None)
        if layers is None:
            raise ValueError("base model exposes no `.layers` for last_n freezing")
        for layer in layers[-last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        return
    raise ValueError(f"Unknown freezing mode: {mode!r}")


def _evaluate_transformer(model, loader, device, criterion) -> tuple[list, list, float]:
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch).logits.float()
            total_loss += criterion(logits, labels).item() * len(labels)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
    return y_true, y_pred, total_loss / len(loader.dataset)


def train_transformer(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    label_names: list[str],
    *,
    model_name: str = MODEL_NAME,
    max_length: int = 1024,
    epochs: int = 4,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    pooling: str = "cls",
    freeze: str = "full",
    last_n: int = 2,
    use_class_weights: bool = True,
    patience: int = 2,
    seed: int = 42,
    save_dir: str | Path | None = None,
    verbose: bool = True,
):
    """Fine-tune ModernBERT for sequence classification.

    Returns ``(model, tokenizer, history)``. ``pooling`` selects the HF
    sequence-classifier pooling strategy ('cls' or 'mean').
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    set_seed(seed)
    device = get_device()
    num_classes = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
        classifier_pooling=pooling,
    ).to(device)
    _apply_freezing(model, freeze, last_n)
    if verbose:
        print(
            f"ModernBERT: {count_parameters(model):,} trainable params "
            f"(freeze={freeze}, pooling={pooling}, max_length={max_length})"
        )

    train_ds = TransformerTextDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = TransformerTextDataset(val_texts, val_labels, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    weights = (
        compute_class_weights(train_labels, num_classes).to(device)
        if use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps
    )

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(
            tqdm(train_loader, disable=not verbose, desc=f"epoch {epoch}")
        ):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch).logits.float()
            loss = criterion(logits, labels) / grad_accum
            loss.backward()
            running += loss.item() * grad_accum * len(labels)
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        train_loss = running / len(train_ds)

        y_true, y_pred, val_loss = _evaluate_transformer(
            model, val_loader, device, criterion
        )
        m = compute_metrics(y_true, y_pred, label_names)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": m["macro_f1"],
                "val_accuracy": m["accuracy"],
                "seconds": round(time.time() - t0, 1),
            }
        )
        if verbose:
            print(
                f"epoch {epoch}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}"
                f"  val_macroF1 {m['macro_f1']:.4f}  val_acc {m['accuracy']:.4f}"
            )

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stopping (best macro-F1 {best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2))
        if verbose:
            print(f"saved best model to {save_dir}")
    return model, tokenizer, history





def main() -> None:
    parser = argparse.ArgumentParser(description="Train a website-category classifier.")
    parser.add_argument("model", choices=["scratch", "modernbert"])
    parser.add_argument("--arch", default="cnn", help="scratch arch: cnn | bilstm")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--freeze", default="last_n", help="full | head | last_n")
    parser.add_argument("--limit", type=int, help="cap rows per split (smoke test)")
    args = parser.parse_args()

    train, val, test, label_map = load_processed()
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    if args.limit:
        train, val = train.head(args.limit), val.head(args.limit)
    print(
        f"train={len(train)}  val={len(val)}  test={len(test)}  "
        f"classes={len(label_names)}"
    )

    if args.model == "scratch":
        model, tok, hist = train_scratch_model(
            train["content"].tolist(),
            train["label"].tolist(),
            val["content"].tolist(),
            val["label"].tolist(),
            label_names,
            arch=args.arch,
            epochs=args.epochs or 15,
            lr=args.lr or 1e-3,
            batch_size=max(args.batch_size, 32),
        )
        loader = DataLoader(
            ScratchTextDataset(
                test["content"].tolist(), test["label"].tolist(), tok, 400
            ),
            batch_size=64,
            collate_fn=scratch_collate,
        )
        y_true, y_pred, _ = _evaluate_scratch(
            model, loader, get_device(), nn.CrossEntropyLoss()
        )
        print_report(
            compute_metrics(y_true, y_pred, label_names),
            f"From-scratch ({args.arch}): TEST",
        )
    else:
        model, tok, hist = train_transformer(
            train["content"].tolist(),
            train["label"].tolist(),
            val["content"].tolist(),
            val["label"].tolist(),
            label_names,
            max_length=args.max_length,
            epochs=args.epochs or 4,
            batch_size=args.batch_size,
            lr=args.lr or 3e-5,
            freeze=args.freeze,
            save_dir=ARTIFACTS / "modernbert",
        )
        ds = TransformerTextDataset(
            test["content"].tolist(), test["label"].tolist(), tok, args.max_length
        )
        from transformers import DataCollatorWithPadding

        loader = DataLoader(
            ds, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tok)
        )
        y_true, y_pred, _ = _evaluate_transformer(
            model, loader, get_device(), nn.CrossEntropyLoss()
        )
        print_report(compute_metrics(y_true, y_pred, label_names), "ModernBERT: TEST")


if __name__ == "__main__":
    main()
