"""Figures for the project report, written as PNGs by `overtake.train`."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .features import DEFAULTS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]                            # categorical slots 1-2
BATTERY_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, low -> high charge

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
    "font.size": 9,
    "lines.linewidth": 2,
})


def _percent_axis(ax):
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def training_curve(history, path):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    epochs = [h["epoch"] for h in history]
    for key, label, color in (("train_loss", "Training", SERIES[0]), ("val_loss", "Validation", SERIES[1])):
        ax.plot(epochs, [h[key] for h in history], color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Training loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration(y, curves, path, bins=10):
    """Reliability diagram: predicted probability vs how often the overtake happened."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("Perfect calibration", (0.62, 0.62), rotation=38, color=MUTED,
                xytext=(0, 6), textcoords="offset points", ha="center")
    edges = np.linspace(0, 1, bins + 1)
    for (label, p), color in zip(curves.items(), SERIES):
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        keep = [b for b in range(bins) if (idx == b).sum() >= 20]
        mean_p = [p[idx == b].mean() for b in keep]
        freq = [y[idx == b].mean() for b in keep]
        ax.plot(mean_p, freq, color=color, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _percent_axis(ax)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed overtake rate")
    ax.set_title("Calibration on the test set")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def what_if(predictor, path, base=None):
    """How the probability responds to gap, battery charge, run length and Boost."""
    base = {**DEFAULTS, **(base or {})}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

    gaps = np.linspace(0.05, 2.0, 80)
    for battery, color in zip((25, 50, 75, 100), BATTERY_RAMP):
        p = predictor.predict([{**base, "gap_s": g, "overtake_mode": g <= 1.0, "attacker_battery": battery}
                               for g in gaps])
        left.plot(gaps, p, color=color, label=f"{battery}% battery")
    p_base = predictor.predict(base)
    left.plot([base["gap_s"]], [p_base], marker="o", markersize=8, color=BATTERY_RAMP[-1],
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    left.annotate(f"Default scenario: {p_base:.0%}", (base["gap_s"], p_base), xytext=(10, 10),
                  textcoords="offset points", color=INK)
    left.axvline(1.0, color=AXIS, linewidth=1)
    left.annotate("Overtake Mode\nlimit (1 s)", (1.0, 0.97), xytext=(4, 0), textcoords="offset points",
                  va="top", color=MUTED)
    left.set_xlabel("Gap to the car ahead (s)")
    left.set_ylabel("P(overtake)")
    left.set_title("Gap and attacker battery charge")
    left.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    _percent_axis(left)

    lengths = np.linspace(250, 1300, 80)
    for boost, label, color in ((True, "With Boost", SERIES[0]), (False, "Without Boost", SERIES[1])):
        p = predictor.predict([{**base, "run_length_m": L, "attacker_boost": boost} for L in lengths])
        right.plot(lengths, p, color=color, label=label)
        right.annotate(label, (lengths[-1], p[-1]), xytext=(-4, 8), textcoords="offset points",
                       ha="right", color=INK_SECONDARY)
    right.set_xlabel("Length of the straight (m)")
    right.set_title(f"Straight length, {base['gap_s']:g} s gap")
    right.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
