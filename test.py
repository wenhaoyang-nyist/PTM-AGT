"""Evaluate a saved fold checkpoint.

    python test.py --dataset THCA --seed 42 --fold 1 \
        --checkpoint results/THCA/checkpoints/seed42_fold1.pt
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from model import PTMAGT
from preprocess import (
    COHORTS,
    apply_prior_weighting,
    attention_mask,
    build_train_test_adjacency,
    estimate_threshold,
    load_association,
    load_cohort,
    make_folds,
    shared_protein_pairs,
)
from utils.engine import predict
from utils.metrics import classification_metrics, reported_metrics
from utils.seed import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PTM-AGT fold checkpoint")
    parser.add_argument("--dataset", required=True, choices=list(COHORTS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--adj-parameter", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--keep-first-row", action="store_true")
    parser.add_argument("--save-prob", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cohort = load_cohort(args.root, args.dataset, keep_first_row=args.keep_first_row)
    scores, protein_table = load_association(args.root)
    pairs = shared_protein_pairs(cohort.feature_names, protein_table)

    # rebuild the same split, scaler and graph as during training
    train_index, test_index = make_folds(cohort.y, args.folds, args.seed)[args.fold - 1]

    scaler = RobustScaler()
    x_train = scaler.fit_transform(cohort.x[train_index])
    x_test = scaler.transform(cohort.x[test_index])
    x_train_w, x_test_w = apply_prior_weighting(
        x_train, x_test, pairs, scores,
        disease_column=cohort.disease_column,
        eta=args.eta,
        magnitude_decay=cohort.magnitude_decay,
    )

    x_train_t = torch.as_tensor(x_train_w, dtype=torch.float32, device=device)
    x_test_t = torch.as_tensor(x_test_w, dtype=torch.float32, device=device)
    x_all_t = torch.cat([x_train_t, x_test_t], dim=0)

    threshold = estimate_threshold(args.adj_parameter, x_train_t)
    mask_all = attention_mask(build_train_test_adjacency(x_all_t, x_train_t.shape[0], threshold))

    model = PTMAGT(
        in_dim=x_train_w.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=cohort.num_classes,
        dropout=args.dropout,
        residual_scale=cohort.alpha,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"])

    prob = predict(model, x_all_t, mask_all, x_train_t.shape[0])
    report = reported_metrics(classification_metrics(cohort.y[test_index], prob), cohort.num_classes)

    print(f"{args.dataset} seed={args.seed} fold={args.fold}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in report.items()))

    if args.save_prob:
        out_path = os.path.splitext(args.checkpoint)[0] + "_prob.csv"
        with open(out_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["y_true", *[f"p_class{c}" for c in range(cohort.num_classes)]])
            for label, row in zip(cohort.y[test_index], prob):
                writer.writerow([int(label), *[f"{p:.6f}" for p in row]])
        print(f"  probabilities written to {out_path}")


if __name__ == "__main__":
    main()
