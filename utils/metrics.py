"""Metrics reported in the paper."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def class_prevalence_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Weight of a sample = frequency of its class in the training fold (N_c / N)."""
    y = np.asarray(y).reshape(-1)
    counts = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        counts[c] = np.sum(y == c)
    weights = np.zeros(y.shape, dtype=float)
    for c in range(num_classes):
        weights[np.where(y == c)[0]] = counts[c] / np.sum(counts)
    return weights


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = prob.argmax(1)
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1": np.nan,
        "auc": np.nan,
    }
    if prob.shape[1] == 2:
        metrics["f1"] = f1_score(y_true, y_pred)
        metrics["auc"] = roc_auc_score(y_true, prob[:, 1])
    return metrics


def reported_metrics(metrics: dict, num_classes: int) -> dict:
    """Binary cohorts: ACC / F1 / AUC. KIPAN: ACC / weighted F1 / macro F1."""
    if num_classes == 2:
        return {"ACC": metrics["acc"], "F1": metrics["f1"], "AUC": metrics["auc"]}
    return {"ACC": metrics["acc"], "F1_w": metrics["f1_weighted"], "F1_m": metrics["f1_macro"]}
