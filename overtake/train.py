"""Train the overtake-probability network and evaluate it against the alternatives.

    python -m overtake.train                         # train on simulated attempts
    python -m overtake.train --data attempts.csv     # train on your own labelled attempts

A CSV needs every column in `overtake.features.FEATURES` plus an ``overtake``
column (1 if the move stuck, 0 if not). By default the trained ensemble is
written to ``models/overtake_net.pt``, the test metrics to
``reports/results/test_metrics.json`` and the training-curve, calibration and
what-if figures to ``reports/figures/``.

Data are split 80/10/10 into training, validation and test sets. The
validation set is used for early stopping and the learning-rate schedule
(and, in `overtake.experiments`, for choosing the architecture); the test set
is used only once, for the final numbers.
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
HIDDEN = (64, 64)  # chosen by experiments.selection (one-standard-error rule)
# Networks averaged. experiments.ablation compared one network with an ensemble of five: the ensemble's
# validation error against the exact probability was about 10 % lower. No other sizes were tried.
ENSEMBLE = 5


def load_data(path=None, samples=300_000, seed=0):
    """Read labelled attempts from a CSV, or simulate them.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV of labelled attempts: every `overtake.features.FEATURES` column
        plus a 0/1 ``overtake`` label. Omit to simulate.
    samples : int, optional
        Attempts to simulate (300,000); ignored when ``path`` is given.
    seed : int, optional
        Seed for the simulated scenarios and outcomes; ignored when ``path``
        is given.

    Returns
    -------
    pandas.DataFrame
        One row per attempt: the schema columns and ``overtake``. Simulated
        data also has ``p_true``, the exact probability from quadrature (for
        validation only, never a model input). Extra CSV columns are kept.

    Raises
    ------
    FileNotFoundError
        If there is no file at ``path`` (from `pandas.read_csv`).
    ValueError
        If the CSV is missing columns, has values outside the schema (see
        `overtake.features.validate`), or has labels other than 0/1.
    """
    if path is None:
        return simulator.generate_dataset(samples, seed=seed)
    df = pd.read_csv(path)
    validate(df)
    if LABEL not in df or not df[LABEL].isin([0, 1]).all():
        raise ValueError(f"{path}: needs an '{LABEL}' column of 0/1 values")
    return df


def split(df, seed=0):
    """Shuffle the rows and split them into training (80 %), validation (10 %) and test (10 %) sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Labelled attempts, one per row.
    seed : int, optional
        Seeds the shuffle; the same seed always gives the same split.

    Returns
    -------
    train_df, val_df, test_df : pandas.DataFrame
        Validation and test get ``len(df) // 10`` rows each (30,000 of
        300,000); training gets the rest (80 %, plus the remainder when
        ``len(df)`` is not a multiple of 10). With fewer than 10 rows the
        validation and test sets are empty. Rows keep their original index.
    """
    idx = np.random.default_rng(seed).permutation(len(df))
    n = len(df) // 10
    # Validation and test are the first two tenths of the shuffled order; training keeps the rest.
    return df.iloc[idx[2 * n:]], df.iloc[idx[:n]], df.iloc[idx[n:2 * n]]


def train_network(train_df, val_df, hidden=HIDDEN, dropout=0.1, epochs=100, batch_size=1024, lr=2e-3,
                  patience=5, seed=0, physics_features=False, log=print):
    """Train an `OvertakeNet` with binary cross-entropy and early stopping.

    Parameters
    ----------
    train_df, val_df : pandas.DataFrame
        Training and validation attempts: the schema columns plus the 0/1
        ``overtake`` label. Both must be non-empty.
    hidden : tuple of int, optional
        Hidden layer widths (`HIDDEN` = (64, 64)).
    dropout : float, optional
        Probability of zeroing each hidden unit during training (0.1); off
        at prediction.
    epochs : int, optional
        Maximum number of passes over the training data (100); must be at
        least 1.
    batch_size : int, optional
        Attempts per AdamW step (1,024; the last batch of an epoch may be
        smaller).
    lr : float, optional
        Initial AdamW learning rate (2e-3). ``ReduceLROnPlateau`` halves it
        after 3 consecutive epochs in which the validation loss fails to fall
        by more than 0.01 % of its best value (``patience=2`` is the number of
        such epochs tolerated before it acts), and again after every further
        3 such epochs.
    patience : int, optional
        Stop after this many consecutive epochs (5) in which the validation
        loss does not fall more than 1e-5 below its best. The best weights
        are restored whether or not training stops early.
    seed : int, optional
        Seeds weight initialisation, dropout and batch order, so runs are
        repeatable.
    physics_features : bool, optional
        Include the derived energy-cover inputs (`overtake.features.DERIVED`);
        off in the final model (see `overtake.experiments.ablation`).
    log : callable, optional
        Receives one progress line per epoch, plus one more if early
        stopping ends training.

    Returns
    -------
    predictor : OvertakePredictor
        The network from the best epoch (lowest validation loss, to within
        the 1e-5 improvement threshold) and its fitted preprocessing.
    history : list of dict
        One dict per epoch run: ``epoch`` (from 1); ``train_loss`` (mean
        binary cross-entropy over the epoch's batches, dropout on);
        ``val_loss`` (binary cross-entropy on ``val_df`` after the epoch,
        dropout off); ``seconds`` (wall-clock time of the epoch, validation
        included).

    Raises
    ------
    ValueError
        If either table fails `overtake.features.validate`.
    """
    # Seeds the torch generator behind weight initialisation, dropout masks and the batch order.
    torch.manual_seed(seed)
    # Statistics from the training set only, so validation and test rows are scaled exactly like unseen data.
    preprocessor = Preprocessor(physics_features=physics_features).fit(train_df)
    x_cat, x_num = preprocessor.transform(train_df)
    y = torch.tensor(train_df[LABEL].to_numpy(dtype=np.float32))
    v_cat, v_num = preprocessor.transform(val_df)
    v_y = torch.tensor(val_df[LABEL].to_numpy(dtype=np.float32))

    model = OvertakeNet(hidden, dropout, preprocessor.n_inputs)
    # AdamW (Loshchilov and Hutter) applies weight decay separately from the adaptive step. The
    # learning rate, weight decay (1e-4), dropout and batch size are fixed starting values; the
    # experiments in `overtake.experiments` do not tune them.
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Halve the step size once validation loss has stalled for 3 epochs (patience=2 tolerated, the 3rd
    # acts). PyTorch's default relative threshold (1e-4) decides what counts as a stall; early stopping
    # below uses its own 1e-5 absolute test.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    # The network outputs logits. BCEWithLogitsLoss puts the sigmoid inside the loss in log-sum-exp
    # form, so it stays exact for large |logit|. A separate sigmoid rounds to 1.0 in float32 (a logit
    # of 20 already does), after which BCELoss can only return its clamp value of 100.
    loss_fn = nn.BCEWithLogitsLoss()

    history, best_loss, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        # New random batch order every epoch, reproducible through torch.manual_seed above.
        order = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch_size):
            batch = order[i:i + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(x_cat[batch], x_num[batch]), y[batch])
            loss.backward()
            optimiser.step()
            # Weight by batch size: the last batch is smaller, so total / len(y) is the mean loss per attempt.
            total += loss.item() * len(batch)
        # Dropout off for validation; the whole validation set goes through in one pass.
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(v_cat, v_num), v_y).item()
        scheduler.step(val_loss)
        seconds = time.perf_counter() - start
        # train_loss is averaged over the epoch with dropout on while the weights are still changing,
        # so it can sit above val_loss, especially in early epochs. That is expected.
        history.append({"epoch": epoch, "train_loss": total / len(y), "val_loss": val_loss, "seconds": seconds})
        log(f"epoch {epoch:3d}  train {total / len(y):.4f}  val {val_loss:.4f}  ({seconds:.1f}s)")

        # Only a fall of more than 1e-5 counts as an improvement, so negligible changes in the
        # loss do not reset the patience counter.
        if val_loss < best_loss - 1e-5:
            # deepcopy: state_dict() returns references to the live tensors, which later optimiser
            # steps would overwrite.
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                log(f"validation loss stopped improving; keeping epoch {epoch - stale}")
                break
    # Restore the best epoch's weights, whether the loop ended early or ran all the epochs.
    model.load_state_dict(best_state)
    return OvertakePredictor(model, preprocessor), history


def train_ensemble(train_df, val_df, n_members=ENSEMBLE, seed=0, log=print, **kwargs):
    """Train ``n_members`` networks with seeds ``seed, seed + 1, ...`` and average them.

    Each member is trained exactly as by `train_network`; their predicted
    probabilities are averaged, which reduces the run-to-run variation of a
    single network.

    Parameters
    ----------
    train_df, val_df : pandas.DataFrame
        As for `train_network`; every member sees the same data.
    n_members : int, optional
        Number of networks (`ENSEMBLE` = 5); must be at least 1.
    seed : int, optional
        Member k (counting from 0) is trained with seed ``seed + k``.
    log : callable, optional
        Receives one line per member and the members' per-epoch lines.
    **kwargs
        Passed to `train_network` (for example ``epochs``, ``batch_size``,
        ``lr``, ``hidden``).

    Returns
    -------
    predictor : OvertakePredictor
        All members, sharing one preprocessor (they see the same training data).
    histories : list of list of dict
        Each member's training history (see `train_network`).

    Raises
    ------
    ValueError
        If either table fails `overtake.features.validate`.

    Examples
    --------
    >>> from overtake import simulator, train
    >>> df = simulator.generate_dataset(20_000, seed=0)
    >>> train_df, val_df, test_df = train.split(df)
    >>> predictor, histories = train.train_ensemble(  # doctest: +SKIP
    ...     train_df, val_df, n_members=2, log=lambda _: None)
    >>> predictor.predict(test_df).shape  # doctest: +SKIP
    (2000,)
    """
    members, histories = [], []
    for k in range(n_members):
        log(f"-- network {k + 1} of {n_members}")
        predictor, history = train_network(train_df, val_df, seed=seed + k, log=log, **kwargs)
        members.append(predictor)
        histories.append(history)
    # Every member fitted an identical Preprocessor on the same train_df (fit is deterministic),
    # so the first one serves the whole ensemble.
    return OvertakePredictor([m.model for m in members], members[0].preprocessor), histories


def expected_calibration_error(y, p, bins=10):
    """Expected calibration error (ECE, as in Guo et al. 2017) over equal-width probability bins.

    Predictions are grouped into ``bins`` bins of width ``1 / bins``. The
    result is the sum over non-empty bins of (share of predictions in the bin)
    x |mean predicted probability - observed overtake rate|.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Observed 0/1 outcomes.
    p : numpy.ndarray, shape (n,)
        Predicted probabilities in [0, 1].
    bins : int, optional
        Number of equal-width bins (10).

    Returns
    -------
    float
        ECE as a probability; 0 means perfectly calibrated (x 100 for
        percentage points).
    """
    # np.digitize gives 1..bins for p in [0, 1) and bins + 1 for p == 1; shift to 0-based and fold
    # p == 1 into the last bin.
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    # (idx == b).mean() is the bin's share of all predictions, the ECE weight.
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p, p_true=None):
    """Evaluation metrics for predicted probabilities.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Observed 0/1 outcomes. It should contain both classes; otherwise
        ``roc_auc`` is undefined (NaN with a warning on the installed
        scikit-learn, a ``ValueError`` on older versions).
    p : numpy.ndarray, shape (n,)
        Predicted probabilities; clipped to [1e-7, 1 - 1e-7] before every
        metric so log loss stays finite.
    p_true : numpy.ndarray, shape (n,), optional
        True probabilities (simulated data only).

    Returns
    -------
    dict of str -> float
        ``log_loss``, ``brier``, ``roc_auc``, ``accuracy`` (threshold 0.5),
        ``calibration_error`` (`expected_calibration_error`, 10 bins) and,
        with ``p_true``, ``mae_vs_true`` and ``max_error_vs_true``: the mean
        and largest abs(p - p_true), as probabilities.
    """
    # Keep log loss finite if a model outputs exactly 0 or 1; moves p by at most 1e-7.
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
    """Command-line entry point: train the models, score them once on the test set and save the results.

    Loads or simulates the attempts (`load_data`), splits them 80/10/10
    (`split`), trains ``--ensemble`` networks (`train_ensemble`) and the
    logistic-regression and gradient-boosting baselines on the training set,
    then evaluates all three on the test set (`metrics`) and prints the
    results as JSON.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without the program name; ``sys.argv[1:]`` if
        omitted. See ``python -m overtake.train --help``.

    Raises
    ------
    SystemExit
        Status 2 if an option is malformed (argparse).
    FileNotFoundError
        If ``--data`` names a file that does not exist.
    ValueError
        If ``--data`` fails the schema or label checks (see `load_data`).

    Notes
    -----
    Writes five files: ``--out`` (default ``models/overtake_net.pt``: weights,
    preprocessing, test metrics, the first member's history and the total fit
    time) and, under ``--reports`` (default ``reports/``),
    ``results/test_metrics.json`` and ``figures/training_curve.png``,
    ``figures/calibration.png`` and ``figures/what_if.png``. The README gives
    about 4 minutes with the defaults.

    Examples
    --------
    A quick run on 20,000 simulated attempts with a single network::

        python -m overtake.train --samples 20000 --ensemble 1 --out tmp/net.pt --reports tmp/reports
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help="CSV of labelled attempts (default: simulate)")
    parser.add_argument("--samples", type=int, default=300_000, help="simulated attempts to generate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=ENSEMBLE, help="networks to train and average")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    df = load_data(args.data, args.samples, args.seed)
    print(f"{len(df):,} attempts ({time.perf_counter() - start:.0f}s); overtake rate {df[LABEL].mean():.1%}")
    train_df, val_df, test_df = split(df, args.seed)

    predictor, histories = train_ensemble(train_df, val_df, args.ensemble, seed=args.seed, epochs=args.epochs,
                                          batch_size=args.batch_size, lr=args.lr)
    logistic = baselines.logistic_regression().fit(train_df)
    boosting = baselines.gradient_boosting(args.seed).fit(train_df)

    y = test_df[LABEL].to_numpy()
    # p_true (the exact quadrature probability) exists only for simulated data; a real CSV has no ground truth.
    p_true = test_df["p_true"].to_numpy() if "p_true" in test_df else None
    predictions = {"Neural network": predictor.predict(test_df),
                   logistic.name: logistic.predict(test_df),
                   boosting.name: boosting.predict(test_df)}
    results = {"test_attempts": len(test_df),
               "models": {name: metrics(y, p, p_true) for name, p in predictions.items()}}

    # Total time of the epoch loops for all members; excludes data generation, baselines and plots.
    fit_seconds = sum(h["seconds"] for history in histories for h in history)
    # Only the first member's history is saved and plotted; the others stay in `histories`.
    predictor.save(args.out, metrics=results, history=histories[0], fit_seconds=fit_seconds)
    (args.reports / "figures").mkdir(parents=True, exist_ok=True)
    (args.reports / "results").mkdir(parents=True, exist_ok=True)
    (args.reports / "results" / "test_metrics.json").write_text(json.dumps(results, indent=2))
    plots.training_curve(histories[0], args.reports / "figures" / "training_curve.png")
    plots.calibration(y, predictions, args.reports / "figures" / "calibration.png")
    # exact(scenarios) gives the true P(overtake) from the simulator's quadrature, which the what-if
    # figure draws as dashed lines. A user CSV has no known truth, so the figure shows the network alone.
    exact = (lambda s: simulator.probability(pd.DataFrame(s))) if args.data is None else None
    plots.what_if(predictor.predict, args.reports / "figures" / "what_if.png", exact=exact)

    print(json.dumps(results, indent=2))
    print(f"saved model to {args.out} and results to {args.reports}/")


if __name__ == "__main__":
    main()
