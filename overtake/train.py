"""Train the overtake-probability network and evaluate it against the alternatives.

    python -m overtake.train                         # train on simulated attempts
    python -m overtake.train --data attempts.csv     # train on your own labelled attempts

A CSV needs every column in `overtake.features.FEATURES` plus an ``overtake``
column (1 if the move stuck, 0 if not). The trained network is written to
``models/`` and metrics and figures to ``reports/``.

Data are split 80/10/10 into training, validation and test sets. The
validation set is used for early stopping (and, in `overtake.experiments`,
for choosing the architecture); the test set is used only once, for the final
numbers.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

from . import baselines, plots, simulator
from .features import LABEL, Preprocessor, validate
from .model import OvertakeNet
from .predict import DEFAULT_MODEL, OvertakePredictor

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
HIDDEN = (128, 128, 64)  # chosen in experiments.model_selection


def load_data(path=None, samples=300_000, seed=0):
    """Read labelled attempts from a CSV, or simulate them.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV with the schema columns and an ``overtake`` label.
    samples : int, optional
        Number of attempts to simulate when ``path`` is not given.
    seed : int, optional

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the CSV is missing columns, has values outside the schema, or has
        labels other than 0/1.
    """
    if path is None:
        return simulator.generate_dataset(samples, seed=seed)
    df = pd.read_csv(path)
    validate(df)
    if LABEL not in df or not df[LABEL].isin([0, 1]).all():
        raise ValueError(f"{path}: needs an '{LABEL}' column of 0/1 values")
    return df


def split(df, seed=0):
    """Shuffle and split into training (80 %), validation (10 %) and test (10 %) sets."""
    idx = np.random.default_rng(seed).permutation(len(df))
    n = len(df) // 10
    return df.iloc[idx[2 * n:]], df.iloc[idx[:n]], df.iloc[idx[n:2 * n]]


def train_network(train_df, val_df, hidden=HIDDEN, dropout=0.1, epochs=40, batch_size=1024, lr=2e-3,
                  patience=5, seed=0, log=print):
    """Train an `OvertakeNet` with binary cross-entropy and early stopping.

    Parameters
    ----------
    train_df, val_df : pandas.DataFrame
        Training and validation attempts with the ``overtake`` label.
    hidden : tuple of int, optional
        Hidden layer widths.
    dropout : float, optional
    epochs : int, optional
        Maximum number of passes over the training data.
    batch_size : int, optional
    lr : float, optional
        Initial AdamW learning rate; halved when validation loss plateaus for 2 epochs.
    patience : int, optional
        Stop after this many epochs without validation improvement and restore the best weights.
    seed : int, optional
        Seeds weight initialisation and batch order, so runs are repeatable.
    log : callable, optional
        Receives one progress line per epoch.

    Returns
    -------
    predictor : OvertakePredictor
        The best network (lowest validation loss) and its preprocessing.
    history : list of dict
        Per-epoch training and validation loss and wall-clock time.
    """
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
        start = time.perf_counter()
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
        seconds = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": total / len(y), "val_loss": val_loss, "seconds": seconds})
        log(f"epoch {epoch:3d}  train {total / len(y):.4f}  val {val_loss:.4f}  ({seconds:.1f}s)")

        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                log(f"validation loss stopped improving; keeping epoch {epoch - stale}")
                break
    model.load_state_dict(best_state)
    return OvertakePredictor(model, preprocessor), history


def expected_calibration_error(y, p, bins=10):
    """Average gap between predicted and observed rates over 10 equal-width probability bins."""
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p, p_true=None):
    """Evaluation metrics for predicted probabilities.

    Parameters
    ----------
    y : numpy.ndarray
        Observed 0/1 outcomes.
    p : numpy.ndarray
        Predicted probabilities.
    p_true : numpy.ndarray, optional
        True probabilities (simulated data only).

    Returns
    -------
    dict
        ``log_loss``, ``brier``, ``roc_auc``, ``accuracy`` (threshold 0.5),
        ``calibration_error`` and, with ``p_true``, ``mae_vs_true`` and
        ``max_error_vs_true``.
    """
    p = np.clip(p, 1e-7, 1 - 1e-7)
    out = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "calibration_error": expected_calibration_error(y, p),
    }
    if p_true is not None:
        out["mae_vs_true"] = float(np.abs(p - p_true).mean())
        out["max_error_vs_true"] = float(np.abs(p - p_true).max())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help="CSV of labelled attempts (default: simulate)")
    parser.add_argument("--samples", type=int, default=300_000, help="simulated attempts to generate")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    df = load_data(args.data, args.samples, args.seed)
    print(f"{len(df):,} attempts ({time.perf_counter() - start:.0f}s); overtake rate {df[LABEL].mean():.1%}")
    train_df, val_df, test_df = split(df, args.seed)

    predictor, history = train_network(train_df, val_df, epochs=args.epochs, batch_size=args.batch_size,
                                       lr=args.lr, seed=args.seed)
    logistic = baselines.logistic_regression().fit(train_df)
    boosting = baselines.gradient_boosting(args.seed).fit(train_df)

    y = test_df[LABEL].to_numpy()
    p_true = test_df["p_true"].to_numpy() if "p_true" in test_df else None
    predictions = {"Neural network": predictor.predict(test_df),
                   logistic.name: logistic.predict(test_df),
                   boosting.name: boosting.predict(test_df)}
    results = {"test_attempts": len(test_df),
               "models": {name: metrics(y, p, p_true) for name, p in predictions.items()}}

    predictor.save(args.out, metrics=results, history=history)
    (args.reports / "figures").mkdir(parents=True, exist_ok=True)
    (args.reports / "results").mkdir(parents=True, exist_ok=True)
    (args.reports / "results" / "test_metrics.json").write_text(json.dumps(results, indent=2))
    plots.training_curve(history, args.reports / "figures" / "training_curve.png")
    plots.calibration(y, predictions, args.reports / "figures" / "calibration.png")
    exact = (lambda s: simulator.probability(pd.DataFrame(s))) if args.data is None else None
    plots.what_if(predictor.predict, args.reports / "figures" / "what_if.png", exact=exact)

    print(json.dumps(results, indent=2))
    print(f"saved model to {args.out} and results to {args.reports}/")


if __name__ == "__main__":
    main()
