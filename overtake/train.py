"""Train the overtake-probability network and evaluate it.

    python -m overtake.train                         # train on simulated attempts
    python -m overtake.train --data attempts.csv     # train on your own labelled attempts

A CSV needs every column in `features.FEATURES` plus an `overtake` column (1 if
the move stuck, 0 if not). Results go to `models/` and `reports/`.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from . import plots, simulator
from .features import BINARY, CATEGORICAL, LABEL, NUMERIC, Preprocessor, validate
from .model import OvertakeNet
from .predict import DEFAULT_MODEL, OvertakePredictor

ROOT = Path(__file__).resolve().parent.parent


def split(df, seed):
    """80% train, 10% validation, 10% test."""
    idx = np.random.default_rng(seed).permutation(len(df))
    n = len(df) // 10
    return df.iloc[idx[2 * n:]], df.iloc[idx[:n]], df.iloc[idx[n:2 * n]]


def train_network(train_df, val_df, epochs=40, batch_size=1024, lr=2e-3, patience=5, seed=0,
                  hidden=(128, 128, 64), dropout=0.1, log=print):
    torch.manual_seed(seed)
    preprocessor = Preprocessor().fit(train_df)
    x_cat, x_num = preprocessor.transform(train_df)
    y = torch.tensor(train_df[LABEL].to_numpy(dtype=np.float32))
    v_cat, v_num = preprocessor.transform(val_df)
    v_y = torch.tensor(val_df[LABEL].to_numpy(dtype=np.float32))

    model = OvertakeNet(hidden, dropout)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()

    history, best_loss, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch_size):
            batch = order[i:i + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(x_cat[batch], x_num[batch]), y[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(v_cat, v_num), v_y).item()
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": total / len(y), "val_loss": val_loss})
        log(f"epoch {epoch:3d}  train {total / len(y):.4f}  val {val_loss:.4f}  ({time.time() - start:.1f}s)")

        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                log(f"validation loss stopped improving; keeping epoch {epoch - stale}")
                break
    model.load_state_dict(best_state)
    return OvertakePredictor(model, preprocessor), history


def train_baseline(train_df):
    """Logistic regression on the same inputs: what a model without interactions achieves."""
    columns = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ("numeric", StandardScaler(), NUMERIC),
        ("binary", "passthrough", BINARY),
    ])
    model = make_pipeline(columns, LogisticRegression(max_iter=2000))
    x = train_df[list(CATEGORICAL) + NUMERIC + BINARY].astype({c: int for c in BINARY})
    model.fit(x, train_df[LABEL])
    return lambda df: model.predict_proba(df[x.columns].astype({c: int for c in BINARY}))[:, 1]


def expected_calibration_error(y, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "calibration_error": expected_calibration_error(y, p),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help="CSV of labelled attempts (default: simulate)")
    parser.add_argument("--samples", type=int, default=300_000, help="simulated attempts to generate")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)

    if args.data:
        df = pd.read_csv(args.data)
        validate(df)
        print(f"loaded {len(df):,} attempts from {args.data}")
    else:
        df = simulator.generate_dataset(args.samples, seed=args.seed)
        print(f"simulated {len(df):,} attempts")
    print(f"overtake rate: {df[LABEL].mean():.1%}")

    train_df, val_df, test_df = split(df, args.seed)
    predictor, history = train_network(train_df, val_df, args.epochs, args.batch_size, args.lr, seed=args.seed)
    baseline = train_baseline(train_df)

    y = test_df[LABEL].to_numpy()
    p_net, p_base = predictor.predict(test_df), baseline(test_df)
    results = {
        "test_attempts": len(test_df),
        "network": metrics(y, p_net),
        "logistic_regression": metrics(y, p_base),
    }

    if not args.data:
        # The simulator knows the true probability of each situation, so we can
        # check how close the network gets to it, not just to noisy 0/1 outcomes.
        sample = test_df.iloc[:2000]
        truth = simulator.true_probability(sample, n_runs=400)
        results["mean_abs_error_vs_true_probability"] = {
            "network": float(np.abs(predictor.predict(sample) - truth).mean()),
            "logistic_regression": float(np.abs(baseline(sample) - truth).mean()),
        }

    predictor.save(args.out, metrics=results, history=history)
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "metrics.json").write_text(json.dumps(results, indent=2))
    plots.training_curve(history, args.reports / "training_curve.png")
    plots.calibration(y, {"Neural network": p_net, "Logistic regression": p_base},
                      args.reports / "calibration.png")
    plots.what_if(predictor, args.reports / "what_if.png")

    print(json.dumps(results, indent=2))
    print(f"saved model to {args.out} and figures to {args.reports}/")


if __name__ == "__main__":
    main()
