"""Figures for the project documentation, written as PNGs.

Colours follow a colour-vision-deficiency-checked palette: up to three
categorical series (blue, orange, aqua) and a single-hue blue ramp for ordered
values such as battery charge. Every multi-series figure has a legend.
"""
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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                  # categorical slots 1-3
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

MARKER = {"marker": "o", "markersize": 5, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def _percent_axis(ax, axis="y"):
    getattr(ax, f"set_{axis}lim")(0, 1)
    getattr(ax, f"{axis}axis").set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def _log_ticks(ax, values, axis="x"):
    """Label a log axis at exactly ``values`` in plain notation, without minor labels."""
    target = getattr(ax, f"{axis}axis")
    target.set_major_locator(matplotlib.ticker.FixedLocator(values))
    target.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in values]))
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_curve(history, path):
    """Training and validation loss per epoch.

    Parameters
    ----------
    history : list of dict
        From `overtake.train.train_network`.
    path : str or pathlib.Path
        Output PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4))
    epochs = [h["epoch"] for h in history]
    for key, label, color in (("train_loss", "Training", SERIES[0]), ("val_loss", "Validation", SERIES[1])):
        ax.plot(epochs, [h[key] for h in history], color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Training loss")
    ax.legend(loc="upper right")
    _save(fig, path)


def calibration(y, curves, path, bins=10):
    """Reliability diagram: predicted probability against observed overtake rate.

    Parameters
    ----------
    y : numpy.ndarray
        Observed 0/1 outcomes.
    curves : dict of str -> numpy.ndarray
        Predicted probabilities per model (at most three).
    path : str or pathlib.Path
    bins : int, optional
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("Perfect calibration", (0.62, 0.62), rotation=38, color=MUTED,
                xytext=(0, 6), textcoords="offset points", ha="center")
    edges = np.linspace(0, 1, bins + 1)
    for (label, p), color in zip(curves.items(), SERIES):
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        keep = [b for b in range(bins) if (idx == b).sum() >= 20]
        ax.plot([p[idx == b].mean() for b in keep], [y[idx == b].mean() for b in keep],
                color=color, label=label, **MARKER)
    _percent_axis(ax, "x")
    _percent_axis(ax)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed overtake rate")
    ax.set_title("Calibration on the test set")
    ax.legend(loc="upper left")
    _save(fig, path)


def what_if(predict, path, exact=None, base=None):
    """How the probability responds to gap, battery charge, straight length and Boost.

    Parameters
    ----------
    predict : callable
        Maps a list of scenario dicts to probabilities (e.g. ``OvertakePredictor.predict``).
    path : str or pathlib.Path
    exact : callable, optional
        Same signature, giving the simulator's true probability; drawn dashed
        for comparison.
    base : dict, optional
        Scenario to vary around (defaults to `overtake.features.DEFAULTS`).
    """
    base = {**DEFAULTS, **(base or {})}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)

    def draw(ax, xs, scenarios, color, label):
        ax.plot(xs, predict(scenarios), color=color, label=label)
        if exact is not None:
            ax.plot(xs, exact(scenarios), color=color, linewidth=1, linestyle="--")

    gaps = np.linspace(0.05, 2.0, 80)
    for battery, color in zip((25, 50, 75, 100), BATTERY_RAMP):
        draw(left, gaps, [{**base, "gap_s": g, "overtake_mode": g <= 1.0, "attacker_battery": battery}
                          for g in gaps], color, f"{battery}% battery")
    p_base = float(np.asarray(predict([base]))[0])
    left.plot([base["gap_s"]], [p_base], marker="o", markersize=8, color=BATTERY_RAMP[-1],
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    left.annotate(f"Default scenario: {p_base:.0%}", (base["gap_s"], p_base), xytext=(10, 0),
                  textcoords="offset points", va="center", color=INK)
    left.axvline(1.0, color=AXIS, linewidth=1)
    left.annotate("Overtake Mode limit (1 s)", (1.0, 0.62), xytext=(6, 0), textcoords="offset points",
                  va="center", color=MUTED)
    left.set_xlabel("Gap to the car ahead (s)")
    left.set_ylabel("P(overtake)")
    left.set_title("Gap and attacker battery charge")
    left.legend(loc="upper right")
    _percent_axis(left)

    lengths = np.linspace(250, 1300, 80)
    for boost, label, color in ((True, "With Boost", SERIES[0]), (False, "Without Boost", SERIES[1])):
        draw(right, lengths, [{**base, "run_length_m": L, "attacker_boost": boost} for L in lengths], color, label)
    right.set_xlabel("Length of the straight (m)")
    right.set_title(f"Straight length, {base['gap_s']:g} s gap")
    right.legend(loc="upper left")
    if exact is not None:
        right.annotate("Solid: neural network\nDashed: simulator truth", (0.02, 0.80),
                       xycoords="axes fraction", ha="left", va="top", color=MUTED)
    _save(fig, path)


def convergence(results, path):
    """Error against time step for Euler and RK4 on problems with analytical solutions.

    Parameters
    ----------
    results : dict
        ``{problem_title: {"dt": [...], "euler": [...], "rk4": [...]}}``.
    path : str or pathlib.Path
    """
    fig, axes = plt.subplots(1, len(results), figsize=(10, 3.8))
    for ax, (title, r) in zip(np.atleast_1d(axes), results.items()):
        dt = np.array(r["dt"])
        for method, label, color, order in (("rk4", "RK4", SERIES[0], 4), ("euler", "Euler", SERIES[1], 1)):
            err = np.array(r[method])
            ax.loglog(dt, err, color=color, label=f"{label} (observed order {r[method + '_order']:.2f})", **MARKER)
            ref = err[-1] * (dt / dt[-1]) ** order
            ax.loglog(dt, ref, color=MUTED, linewidth=1, linestyle="--")
        _log_ticks(ax, dt)
        ax.set_xlabel("Time step (s)")
        ax.set_ylabel("Position error at end (m)")
        ax.set_title(title)
        ax.legend(loc="lower right")
    np.atleast_1d(axes)[0].annotate("Dashed: slopes 1 and 4", (0.03, 0.95), xycoords="axes fraction",
                                    va="top", color=MUTED)
    _save(fig, path)


def timestep(results, path, chosen_dt):
    """Error and cost of the full two-car model against time step.

    Parameters
    ----------
    results : dict
        ``{"dt": [...], method: {"max_error_s": [...], "seconds": [...]}}``.
    path : str or pathlib.Path
    chosen_dt : float
        Step used by the simulator, highlighted.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    dt = np.array(results["dt"])
    for method, label, color in (("rk4", "RK4", SERIES[0]), ("euler", "Euler", SERIES[1])):
        err, sec = np.array(results[method]["max_error_s"]), np.array(results[method]["seconds"])
        left.loglog(dt, err, color=color, label=label, **MARKER)
        right.loglog(sec, err, color=color, label=label, **MARKER)
    i = list(dt).index(chosen_dt)
    for ax, x in ((left, chosen_dt), (right, results["rk4"]["seconds"][i])):
        ax.plot([x], [results["rk4"]["max_error_s"][i]], marker="o", markersize=10, color=SERIES[0],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"chosen: dt = {chosen_dt} s", (x, results["rk4"]["max_error_s"][i]), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center", color=INK)
        ax.axhline(1e-3, color=AXIS, linewidth=1)
    left.annotate("1 ms acceptance limit", (dt.min(), 1e-3), xytext=(0, 4), textcoords="offset points",
                  ha="left", color=MUTED)
    _log_ticks(left, sorted(dt))
    left.set_xlabel("Time step (s)")
    left.set_ylabel("Max deficit error (s)")
    left.set_title("Accuracy of the braking-point deficit")
    left.legend(loc="lower right")
    right.set_xlabel("Run time for the test scenarios (s)")
    right.set_title("Cost against accuracy")
    right.legend(loc="upper right")
    _save(fig, path)


def quadrature(results, path):
    """Quadrature rule convergence and Monte Carlo error.

    Parameters
    ----------
    results : dict
        ``{"rules": {rule: {"evaluations": [...], "max_error": [...]}},
        "monte_carlo": {"runs": [...], "rms_error": [...], "theory": [...]}}``.
    path : str or pathlib.Path
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for (rule, r), color in zip(results["rules"].items(), SERIES):
        err = np.maximum(r["max_error"], 1e-17)
        left.loglog(r["evaluations"], err, color=color, label=rule, **MARKER)
    left.set_xlabel("Integrand evaluations per scenario")
    left.set_ylabel("Max absolute error in P")
    left.set_title("Quadrature rules")
    left.legend(loc="upper right")

    mc = results["monte_carlo"]
    right.loglog(mc["runs"], mc["rms_error"], color=SERIES[0], label="Monte Carlo (measured)", **MARKER)
    right.loglog(mc["runs"], mc["theory"], color=MUTED, linewidth=1, linestyle="--",
                 label="Theory: sqrt(p(1-p)/N)")
    _log_ticks(right, mc["runs"])
    right.set_xlabel("Replays per scenario, N")
    right.set_ylabel("RMS error in P")
    right.set_title("Monte Carlo estimate")
    right.legend(loc="lower left")
    _save(fig, path)


def model_selection(architectures, data_sizes, path):
    """Validation error against network size and against training-set size.

    Parameters
    ----------
    architectures : list of dict
        Each with ``label``, ``parameters`` and ``val_mae_vs_true``.
    data_sizes : list of dict
        Each with ``train_size`` and ``val_mae_vs_true``.
    path : str or pathlib.Path
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    params = [a["parameters"] for a in architectures]
    mae = [100 * a["val_mae_vs_true"] for a in architectures]
    left.semilogx(params, mae, color=SERIES[0], **MARKER)
    for a, x, y in zip(architectures, params, mae):
        left.annotate(a["label"], (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                      color=INK_SECONDARY)
    left.set_xlabel("Trainable parameters")
    left.set_ylabel("Validation error vs truth (points)")
    left.set_title("Network size")
    left.set_ylim(0, max(mae) * 1.3)

    sizes = [d["train_size"] for d in data_sizes]
    right.semilogx(sizes, [100 * d["val_mae_vs_true"] for d in data_sizes], color=SERIES[0], **MARKER)
    _log_ticks(right, sizes)
    right.set_xlabel("Training attempts")
    right.set_title("Training-set size")
    right.set_ylim(0, None)
    _save(fig, path)


def method_comparison(regions, path):
    """Mean error against the true probability per operating region and model.

    Parameters
    ----------
    regions : dict
        ``{model: {region: mae}}`` with the same regions for every model.
    path : str or pathlib.Path
    """
    models = list(regions)
    names = list(regions[models[0]])
    x = np.arange(len(names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (model, color) in enumerate(zip(models, SERIES)):
        values = [100 * regions[model][r] for r in names]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width * 0.92, color=color, label=model)
    ax.set_xticks(x, [n.replace("_", "\n", 1).replace("_", " ") for n in names])
    ax.set_ylabel("Mean error vs truth (points)")
    ax.set_title("Error by operating region (test set)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


READABLE = {
    "gap_s": "Gap", "run_length_m": "Straight length", "attacker_battery": "Attacker battery",
    "defender_battery": "Defender battery", "attacker_tyre_age": "Attacker tyre age",
    "defender_tyre_age": "Defender tyre age", "attacker_skill": "Attacker skill", "defender_skill": "Defender skill",
    "top_speed_delta_kph": "Car top-speed advantage", "pace_delta_s": "Car pace advantage",
    "track_temp_c": "Track temperature", "attacker_boost": "Attacker Boost", "defender_boost": "Defender Boost",
    "overtake_mode": "Overtake Mode", "circuit": "Circuit",
}


def sensitivity(rows, path):
    """One-at-a-time sensitivity: probability range as each input moves across its range.

    Parameters
    ----------
    rows : list of dict
        Each with ``input``, ``low``, ``high`` (labels), ``network`` and
        ``exact`` (pairs of probabilities at the low and high settings),
        sorted by importance.
    path : str or pathlib.Path
    """
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    for i, row in enumerate(reversed(rows)):
        lo, hi = row["network"]
        ax.plot([lo, hi], [i, i], color=SERIES[0], linewidth=3, solid_capstyle="round",
                label="Neural network" if i == 0 else None)
        ax.plot(row["exact"], [i, i], linestyle="none", marker="|", markersize=12, markeredgewidth=2,
                color=SERIES[1], label="Simulator truth" if i == 0 else None)
        # Label each setting on the outside of the bar and its truth marker.
        ends = sorted([(lo, row["exact"][0], row["low"]), (hi, row["exact"][1], row["high"])])
        (l_net, l_true, l_text), (r_net, r_true, r_text) = ends
        ax.annotate(l_text, (min(l_net, l_true), i), xytext=(-7, 0), textcoords="offset points", ha="right",
                    va="center", color=MUTED, fontsize=8)
        ax.annotate(r_text, (max(r_net, r_true), i), xytext=(7, 0), textcoords="offset points", ha="left",
                    va="center", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(rows)), [READABLE.get(r["input"], r["input"]) for r in reversed(rows)])
    _percent_axis(ax, "x")
    ax.set_xlim(-0.12, 1.12)
    ax.set_xlabel("P(overtake)")
    ax.set_title("Sensitivity around the default scenario")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def speed_trace(traces, path):
    """Speed and battery energy of both cars along the straight.

    Parameters
    ----------
    traces : dict of str -> pandas.DataFrame
        Panel title -> output of `overtake.simulator.speed_trace` (at most two).
    path : str or pathlib.Path
    """
    fig, axes = plt.subplots(2, len(traces), figsize=(5 * len(traces), 5.6), sharex="col", squeeze=False)
    for col, (title, tr) in enumerate(traces.items()):
        top, bottom = axes[0, col], axes[1, col]
        for who, label, color in (("attacker", "Attacker", SERIES[0]), ("defender", "Defender", SERIES[1])):
            top.plot(tr["defender_x"], tr[f"{who}_kph"], color=color, label=label)
            bottom.plot(tr["defender_x"], tr[f"{who}_mj"], color=color, label=label)
        top.set_title(title)
        top.set_ylabel("Speed (km/h)")
        bottom.set_ylabel("Battery energy used (MJ)")
        bottom.set_xlabel("Defender distance from corner exit (m)")
        top.legend(loc="lower right")
    _save(fig, path)
