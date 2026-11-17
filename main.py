"""Train and evaluate PTM-AGT on one or more cohorts.

    python main.py --dataset all --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from model import PTMAGT
from preprocess import (
    COHORTS,
    apply_prior_weighting,
    attention_mask,
    build_train_adjacency,
    build_train_test_adjacency,
    estimate_threshold,
    load_association,
    load_cohort,
    make_folds,
    shared_protein_pairs,
)
from utils.engine import train_and_evaluate
from utils.seed import resolve_device, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTM-AGT cancer subtype classification")
    parser.add_argument("--dataset", default="all", choices=["all", *COHORTS])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--results-dir", default=None)

    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--eta", type=float, default=2.0)

    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--adj-parameter", type=int, default=2)

    parser.add_argument("--keep-first-row", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_fold(cohort, pairs, scores, train_index, test_index, fold_index, args, device) -> dict:
    x_train, x_test = cohort.x[train_index], cohort.x[test_index]
    y_train, y_test = cohort.y[train_index], cohort.y[test_index]

    # scaler fitted on the training fold only
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_w, x_test_w = apply_prior_weighting(
        x_train, x_test, pairs, scores,
        disease_column=cohort.disease_column,
        eta=args.eta,
        magnitude_decay=cohort.magnitude_decay,
    )

    x_train_t = torch.as_tensor(x_train_w, dtype=torch.float32, device=device)
    x_test_t = torch.as_tensor(x_test_w, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_all_t = torch.cat([x_train_t, x_test_t], dim=0)

    threshold = estimate_threshold(args.adj_parameter, x_train_t)
    adj_train = build_train_adjacency(x_train_t, threshold)
    adj_all = build_train_test_adjacency(x_all_t, x_train_t.shape[0], threshold)

    model = PTMAGT(
        in_dim=x_train_w.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=cohort.num_classes,
        dropout=args.dropout,
        residual_scale=cohort.alpha,
    ).to(device)

    metrics = train_and_evaluate(
        model=model,
        x_train=x_train_t,
        y_train=y_train_t,
        mask_train=attention_mask(adj_train),
        x_all=x_all_t,
        mask_all=attention_mask(adj_all),
        y_test=y_test,
        num_classes=cohort.num_classes,
        device=device,
        pretrain_epochs=args.pretrain_epochs,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        lr_pretrain=args.lr_pretrain,
        lr_finetune=args.lr_finetune,
        verbose=args.verbose,
    )

    if args.save_checkpoints:
        checkpoint_dir = os.path.join(args.output_dir, cohort.name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "cohort": cohort.name, "seed": args.seed,
             "fold": fold_index, "metrics": metrics},
            os.path.join(checkpoint_dir, f"seed{args.seed}_fold{fold_index}.pt"),
        )

    return metrics


def run_cohort(name: str, args, device, scores, protein_table) -> None:
    cohort = load_cohort(args.root, name, keep_first_row=args.keep_first_row)
    pairs = shared_protein_pairs(cohort.feature_names, protein_table)

    print(
        f"\n=== {name} ===  samples={cohort.x.shape[0]}  features={cohort.x.shape[1]}  "
        f"classes={cohort.num_classes}  shared proteins={len(pairs)}  "
        f"lambda={cohort.magnitude_decay}  alpha={cohort.alpha}"
    )

    set_random_seed(args.seed)
    folds = make_folds(cohort.y, num_folds=args.folds, seed=args.seed)

    rows = []
    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        metrics = run_fold(cohort, pairs, scores, train_index, test_index, fold_index, args, device)
        rows.append(metrics)
        print(f"  fold {fold_index}/{len(folds)}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    keys = list(rows[0].keys())
    summary = {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows]))) for key in keys}

    os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
    csv_path = os.path.join(args.output_dir, name, f"seed{args.seed}_folds.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "fold", *keys])
        writer.writeheader()
        for fold_index, row in enumerate(rows, start=1):
            writer.writerow({"seed": args.seed, "fold": fold_index, **{k: f"{v:.6f}" for k, v in row.items()}})
        writer.writerow({"seed": args.seed, "fold": "mean", **{k: f"{summary[k][0]:.6f}" for k in keys}})
        writer.writerow({"seed": args.seed, "fold": "std", **{k: f"{summary[k][1]:.6f}" for k in keys}})

    config_path = os.path.join(args.output_dir, name, f"seed{args.seed}_config.json")
    with open(config_path, "w") as handle:
        json.dump(vars(args), handle, indent=2)

    print("  mean over folds: " + "  ".join(f"{k}={summary[k][0]:.4f}±{summary[k][1]:.4f}" for k in keys))
    print(f"  written to {csv_path}")


def main() -> None:
    args = parse_args()
    args.output_dir = args.results_dir or os.path.join(args.root, "results")
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    scores, protein_table = load_association(args.root)

    datasets = list(COHORTS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        run_cohort(name, args, device, scores, protein_table)


if __name__ == "__main__":
    main()
