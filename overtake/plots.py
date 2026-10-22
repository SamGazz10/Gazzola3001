"""Figures for the project documentation, written as PNGs.

Colours follow a colour-vision-deficiency-checked palette: up to three
categorical series (blue, orange, aqua) and a single-hue blue ramp for ordered
values such as battery charge. Every multi-series figure has a legend.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def _label_points(ax, xs, ys, labels, marker_pt=4):
    """Label scatter points without labels overlapping each other, the points or the axes edge.

    Points with the most close neighbours are labelled first; each label tries
    positions around its point (right, above, below, left, then further out)
    and takes the first free one. Call after the layout is final.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    radius = marker_pt * fig.dpi / 72
    points = ax.transData.transform(np.column_stack([xs, ys]))
    taken = [matplotlib.transforms.Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)
             for px, py in points]
    taken += [t.get_window_extent(renderer) for t in ax.texts]
    near = [(np.hypot(*(points - pt).T) < 60).sum() for pt in points]
    candidates = [((5, 2), "left", "bottom"), ((5, -2), "left", "top"), ((0, 6), "center", "bottom"),
                  ((0, -6), "center", "top"), ((-5, 2), "right", "bottom"), ((-5, -2), "right", "top"),
                  ((6, 0), "left", "center"), ((-6, 0), "right", "center"),
                  ((0, 14), "center", "bottom"), ((0, -14), "center", "top")]

    def place(i, offset, ha, va):
        text = ax.annotate(labels[i], (xs[i], ys[i]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                           color=INK_SECONDARY, fontsize=8)
        return text, text.get_window_extent(renderer).padded(1)

    for i in sorted(range(len(labels)), key=lambda i: -near[i]):
        for candidate in candidates:
            text, box = place(i, *candidate)
            inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
            if inside and not any(box.overlaps(t) for t in taken):
                break
            text.remove()
        else:  # nowhere free: fall back to the first position
            text, box = place(i, *candidates[0])
        taken.append(box)


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


def stability(results, path):
    """Energy-budget overshoot against time step, Euler and RK4, with the predicted stability limits.

    Parameters
    ----------
    results : dict
        `overtake.experiments.stability` output.
    path : str or pathlib.Path
    """
    deployments = results["deployments"]
    fig, axes = plt.subplots(1, len(deployments), figsize=(10, 3.6), sharey=True)
    dt = np.array(results["dt"])
    for ax, (title, row) in zip(axes, deployments.items()):
        for method, label, color in (("rk4", "RK4", SERIES[0]), ("euler", "Euler", SERIES[1])):
            ax.plot(dt, np.array(row[method]["max_energy_overshoot_j"]) / 1e3, color=color, label=label, **MARKER)
            limit = row["predicted_dt_limit_s"][method]
            ax.axvline(limit, color=color, linewidth=1, linestyle="--")
            ax.annotate(f"{label} limit\n{limit:.2f} s", (limit, 1.0), xycoords=ax.get_xaxis_transform(),
                        xytext=(3, -3), textcoords="offset points", va="top", color=color, fontsize=8)
        ax.axvline(results["chosen_dt"], color=AXIS, linewidth=1)
        ax.annotate(f"chosen\n{results['chosen_dt']} s", (results["chosen_dt"], 0.35),
                    xycoords=ax.get_xaxis_transform(), xytext=(4, 0), textcoords="offset points",
                    ha="left", color=MUTED, fontsize=8)
        ax.set_title(f"{title}: decay rate {row['decay_rate_per_s']:.1f} 1/s")
        ax.set_xlabel("Time step (s)")
    # One legend, in the empty space between the chosen step and the limits.
    axes[-1].legend(loc="upper left", bbox_to_anchor=(0.2, 0.97))
    axes[0].set_ylabel("Energy used beyond the battery (kJ)")
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


def real_telemetry(result, path):
    """Real full-throttle speed trace against the RK4 car model, and the fit over every straight.

    Parameters
    ----------
    result : dict
        ``telemetry`` part of `overtake.realdata.analyse`.
    path : str or pathlib.Path
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    traces = result["traces"]
    models = [k for k in traces if k != "real"]
    for name, color in zip(models, SERIES):
        left.plot(traces[name]["x"], traces[name]["kph"], color=color, label=f"Model: {name}")
    left.plot(traces["real"]["x"], traces["real"]["kph"], color=INK, linewidth=2.5, label="Real (OpenF1)")
    left.set_xlabel("Distance along the straight (m)")
    left.set_ylabel("Speed (km/h)")
    left.set_title(f"Longest straight, car {result['car']['driver_number']}, Monza 2026")
    left.legend(loc="lower right")

    data = [result["rms_kph"][m] for m in models]
    box = right.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                        medianprops={"color": INK, "linewidth": 1.5}, flierprops={"markersize": 3})
    for patch, color in zip(box["boxes"], SERIES):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    right.set_yticks(range(1, len(models) + 1),
                     [f"{m}\nbest fit on {result['best_fit'][m]} of {result['runs']}" for m in models])
    right.set_xlabel("RMS speed error per straight (km/h)")
    right.set_title("All full-throttle straights")
    right.grid(axis="y", visible=False)
    _save(fig, path)


def real_gap(result, path):
    """Pass rate against the gap to the car ahead: real races and the simulator.

    Parameters
    ----------
    result : dict
        ``gap`` part of `overtake.realdata.analyse`.
    path : str or pathlib.Path
    """
    real, sim = result["real"], result["simulated"]
    mids = [(r["low"] + r["high"]) / 2 for r in real]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9))
    ref = real[1]["rate"]
    lo = [r["relative"] - r["ci"][0] / ref for r in real]
    hi = [r["ci"][1] / ref - r["relative"] for r in real]
    left.errorbar(mids, [r["relative"] for r in real], yerr=[lo, hi], color=INK, label="Real 2026 races (per lap)",
                  capsize=3, **MARKER)
    sim_mids = [m for m, q in zip(mids, sim) if q["n"]]
    left.plot(sim_mids, [q["relative"] for q in sim if q["n"]], color=SERIES[1],
              label="Simulator (per attempt)", **MARKER)
    left.axvline(1.0, color=AXIS, linewidth=1)
    left.annotate("Overtake Mode limit", (1.0, 0.5), xycoords=left.get_xaxis_transform(), xytext=(4, 0),
                  textcoords="offset points", rotation=90, va="center", color=MUTED, fontsize=8)
    left.set_xlabel("Gap to the car ahead (s)")
    left.set_ylabel("Pass rate relative to 0.25-0.5 s")
    left.set_title("Shape: how fast the chance falls with gap")
    left.set_ylim(0, None)
    left.legend(loc="upper right")

    rates = [100 * r["rate"] for r in real]
    err = [[100 * (r["rate"] - r["ci"][0]) for r in real], [100 * (r["ci"][1] - r["rate"]) for r in real]]
    right.errorbar(mids, rates, yerr=err, color=INK, capsize=3, **MARKER)
    right.axvline(1.0, color=AXIS, linewidth=1)
    right.set_xlabel("Gap at the start of the lap (s)")
    right.set_ylabel("Laps with an on-track pass (%)")
    right.set_title(f"Real 2026 races ({result['driver_laps']:,} driver-laps, 95 % intervals)")
    right.set_ylim(0, None)
    _save(fig, path)


def real_circuits(result, path):
    """Real on-track passes per lap against the simulator's circuit passability.

    Parameters
    ----------
    result : dict
        `overtake.realdata.analyse` output (uses ``races`` and ``circuits``).
    path : str or pathlib.Path
    """
    rows = [r for r in result["races"] if r.get("passability") is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([r["passability"] for r in rows], [r["on_track_per_lap"] for r in rows], linestyle="none",
            color=SERIES[0], **{**MARKER, "markersize": 7})
    c = result["circuits"]
    ax.annotate(f"Spearman rank correlation {c['spearman_rho']:.2f} (p = {c['p_value']:.3f}, {c['n']} circuits)",
                (0.02, 0.96), xycoords="axes fraction", va="top", color=INK)
    ax.set_xlabel("Simulator passability (estimate, log-odds)")
    ax.set_ylabel("Real on-track passes per race lap")
    ax.set_title("Circuits: model assumption against 2026 races")
    fig.tight_layout()
    fig.canvas.draw()
    _label_points(ax, [r["passability"] for r in rows], [r["on_track_per_lap"] for r in rows],
                  [r["race"].replace("-Francorchamps", "") for r in rows])
    _save(fig, path)


def real_speed_traps(rows, path):
    """Speed-trap readings per race against the 2026 MGU-K speed limits.

    Parameters
    ----------
    rows : list of dict
        ``speed_traps`` part of `overtake.realdata.analyse`.
    path : str or pathlib.Path
    """
    rows = sorted(rows, key=lambda r: r["median"])
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.hlines(y, [r["median"] for r in rows], [r["max"] for r in rows], color=GRID, linewidth=3)
    ax.plot([r["median"] for r in rows], y, linestyle="none", color=SERIES[0], label="Median lap", **MARKER)
    ax.plot([r["p99"] for r in rows], y, linestyle="none", color=SERIES[1], label="99th percentile", **MARKER)
    ax.plot([r["max"] for r in rows], y, linestyle="none", marker="|", markersize=10, markeredgewidth=2,
            color=INK, label="Fastest reading")
    # Labels sit left of their lines, low down, where the slow circuits leave the plot empty.
    for x, row, label in ((337, 0.5, "337 km/h: full power\nends with Overtake Mode"),
                          (355, 1.5, "355 km/h: electric\npower reaches zero")):
        ax.axvline(x, color=AXIS, linewidth=1)
        ax.annotate(label, (x, row), xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                    color=MUTED, fontsize=8)
    ax.set_yticks(y, [r["race"] for r in rows])
    ax.set_xlabel("Highest speed per lap from the timing points (km/h)")
    ax.set_title("Speed traps, every 2026 race lap")
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def workflow_diagram(path):
    """Boxes-and-arrows overview of the computational workflow (top row: the simulator and the network;
    bottom row: what checks each part).

    Colours: blue for the simulator, orange for the network, green for verification and validation.

    Parameters
    ----------
    path : str or pathlib.Path
    """
    # Sized for a full text width of an A4 page, so the text prints at about 7-8 pt.
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    tint = {"sim": "#e6effa", "nn": "#fde9df", "val": "#e3f5ee"}

    def box(x, y, w, h, title, body, kind):
        """Draw a rounded box with a bold title and a smaller description; return its extent."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=tint[kind], edgecolor=AXIS, linewidth=0.8))
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=6.6, color=INK_SECONDARY,
                linespacing=1.3)
        return (x, y, w, h)

    def arrow(a, b, label=None, dy=0.0):
        """Draw an arrow from point ``a`` to point ``b``, optionally labelled at its midpoint."""
        (x0, y0), (x1, y1) = a, b
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     color=INK_SECONDARY, linewidth=0.9))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08 + dy, label, ha="center", va="bottom", fontsize=7,
                    color=INK_SECONDARY)

    # Top row: the simulator and the network; bottom row: what checks each part.
    top, bottom, h = 2.08, 0.12, 1.42
    scen = box(0.05, top, 1.65, h, "Scenario", "19 inputs: gap,\nbattery, Boost, tyres,\nweather, circuit", "sim")
    ode = box(2.0, top, 2.0, h, "Two-car ODE, RK4", "power, traction, drag,\nslipstream, battery;\ndt = 0.05 s, Hermite\nbraking-point crossing", "sim")
    outc = box(4.3, top, 1.75, h, "Outcome model", "time deficit D\n+ execution noise\n+ completion logit", "sim")
    lab = box(6.35, top, 1.55, h, "0/1 labels", "one random\noutcome per\nattempt (the\ntraining data)", "sim")
    net = box(8.2, top, 1.75, h, "Neural network", "5 x MLP 64-64\nwith embeddings,\ntrained on 0/1 only", "nn")
    box(0.05, bottom, 1.65, h, "Real 2026 races", "OpenF1: speed\ntraps, telemetry,\npass rate vs gap", "val")
    verif = box(2.0, bottom, 2.0, h, "Verification", "analytical solutions,\nconvergence order,\nstability, constraints", "val")
    exact = box(4.3, bottom, 1.75, h, "Exact P", "16-point\nGauss-Legendre\nquadrature", "sim")
    valid = box(6.35, bottom, 1.55, h, "Validation", "network vs exact P\nand two baselines\non held-out data", "val")
    pred = box(8.2, bottom, 1.75, h, "Prediction", "P(overtake)\nin about 1 µs\nper scenario", "nn")

    def mid(b, side):
        """Midpoint of the right, left, top or bottom edge of box ``b``."""
        x, y, w, h = b
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]
    arrow(mid(scen, "r"), mid(ode, "l"))
    arrow(mid(ode, "r"), mid(outc, "l"))
    arrow(mid(outc, "r"), mid(lab, "l"))
    arrow(mid(lab, "r"), mid(net, "l"))
    arrow(mid(outc, "b"), mid(exact, "t"))
    arrow(mid(exact, "r"), mid(valid, "l"))
    arrow(mid(net, "b"), mid(pred, "t"))
    arrow((8.2, top), (7.9, bottom + h))           # network -> validation
    arrow(mid(verif, "t"), mid(ode, "b"))           # verification checks the ODE solver
    arrow((1.7, bottom + h), (2.0, top))            # real races check the physics
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

