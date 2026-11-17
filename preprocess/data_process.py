"""Cohort loading, prior weighting and cross-validation splits."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# disease_column: column of the association matrix used for the cohort disease
# alpha: initial AdaptiveLN residual scale, magnitude_decay: lambda of Eq. (1)
COHORTS = {
    "COADREAD": {"num_classes": 2, "disease_column": 22, "alpha": 0.30, "magnitude_decay": 0.1},
    "KIPAN": {"num_classes": 3, "disease_column": 42, "alpha": 0.25, "magnitude_decay": 1.0},
    "THCA": {"num_classes": 2, "disease_column": 51, "alpha": 0.20, "magnitude_decay": 3.0},
}


@dataclass
class Cohort:
    name: str
    x: np.ndarray
    y: np.ndarray
    feature_names: np.ndarray
    num_classes: int
    disease_column: int
    alpha: float
    magnitude_decay: float


def load_cohort(root: str, name: str, keep_first_row: bool = False) -> Cohort:
    """Read one cohort from ``<root>/dataset/<name>/``.

    The expression and label files have no header row, so the default
    ``header=0`` consumes their first row; this reproduces the released numbers
    (488 / 752 / 221 samples). Use ``keep_first_row=True`` to keep that row.
    """
    folder = os.path.join(root, "dataset", name)
    header = None if keep_first_row else 0

    x = pd.read_excel(os.path.join(folder, "samples_protein.xlsx"), header=header).to_numpy().astype(float)
    y = pd.read_excel(os.path.join(folder, "labels.xlsx"), header=header).to_numpy()
    feature_names = pd.read_excel(os.path.join(folder, "feature_name.xlsx"), engine="openpyxl").to_numpy()
    y = np.asarray(y, dtype=float).reshape(-1).astype(int)

    spec = COHORTS[name]
    return Cohort(
        name=name,
        x=x,
        y=y,
        feature_names=feature_names,
        num_classes=spec["num_classes"],
        disease_column=spec["disease_column"],
        alpha=spec["alpha"],
        magnitude_decay=spec["magnitude_decay"],
    )


def load_association(root: str):
    """Return the association score matrix S and the protein table."""
    association_dir = os.path.join(root, "dataset", "association")
    scores = pd.read_excel(
        os.path.join(association_dir, "Association_degree_score_matrix.xlsx"),
        header=None,
        index_col=None,
    ).to_numpy()
    protein_table = pd.read_excel(os.path.join(association_dir, "protein_numbers.xlsx")).to_numpy()
    return scores, protein_table


def shared_protein_pairs(feature_names: np.ndarray, protein_table: np.ndarray) -> np.ndarray:
    """Return [feature_column, association_row] pairs matched on the gene symbol."""
    pairs = [
        [i, j]
        for i in range(feature_names.shape[0])
        for j in range(protein_table.shape[0])
        if feature_names[i, 0] == protein_table[j, 1]
    ]
    return np.asarray(pairs, dtype=int)


def apply_prior_weighting(
    x_train: np.ndarray,
    x_test: np.ndarray,
    pairs: np.ndarray,
    scores: np.ndarray,
    disease_column: int,
    eta: float,
    magnitude_decay: float,
):
    """X_w = X * eta * S[j, r] * exp(-lambda * |X|) on the shared proteins only."""
    x_train_w = x_train.copy()
    x_test_w = x_test.copy()

    for feature_col, association_row in pairs:
        prior = eta * scores[association_row, disease_column]
        x_train_w[:, feature_col] = x_train[:, feature_col] * prior
        x_test_w[:, feature_col] = x_test[:, feature_col] * prior
        x_train_w[:, feature_col] *= np.exp(-magnitude_decay * np.abs(x_train[:, feature_col]))
        x_test_w[:, feature_col] *= np.exp(-magnitude_decay * np.abs(x_test[:, feature_col]))

    return x_train_w, x_test_w


def make_folds(y: np.ndarray, num_folds: int = 5, seed: int = 42):
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))
