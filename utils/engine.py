"""Training and evaluation of one fold."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import classification_metrics, class_prevalence_weights, reported_metrics


def train_epoch(model, optimizer, x, y, attn_mask, sample_weight) -> None:
    model.train()
    optimizer.zero_grad()

    logits = model(x, attn_mask)
    loss = torch.mean(F.cross_entropy(logits, y, reduction="none") * sample_weight)

    loss.backward()
    optimizer.step()


@torch.no_grad()
def predict(model, x_all, attn_mask, num_train: int) -> np.ndarray:
    """Class probabilities of the test rows (the graph contains train + test patients)."""
    model.eval()
    logits = model(x_all, attn_mask)
    return F.softmax(logits[num_train:], dim=1).cpu().numpy()


def train_and_evaluate(
    model,
    x_train,
    y_train,
    mask_train,
    x_all,
    mask_all,
    y_test,
    num_classes: int,
    device,
    pretrain_epochs: int = 500,
    epochs: int = 2500,
    eval_interval: int = 50,
    lr_pretrain: float = 1e-3,
    lr_finetune: float = 5e-4,
    verbose: bool = False,
) -> dict:
    """Warm-up stage, then fine-tuning with the test fold scored every eval_interval epochs."""
    sample_weight = torch.as_tensor(
        class_prevalence_weights(y_train.cpu().numpy(), num_classes),
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_pretrain)
    for _ in range(pretrain_epochs):
        train_epoch(model, optimizer, x_train, y_train, mask_train, sample_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_finetune)
    collected = []
    for epoch in range(epochs + 1):
        train_epoch(model, optimizer, x_train, y_train, mask_train, sample_weight)
        if epoch % eval_interval == 0:
            prob = predict(model, x_all, mask_all, x_train.shape[0])
            metrics = reported_metrics(classification_metrics(y_test, prob), num_classes)
            collected.append(metrics)
            if verbose:
                print(f"    epoch {epoch:5d}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return {key: float(np.mean([c[key] for c in collected])) for key in collected[0]}
