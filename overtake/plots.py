"""Figures for the project documentation, written as PNGs.

Each public function draws one figure from results computed elsewhere, mostly
the dictionaries that `overtake.experiments` saves as JSON in
``reports/results/``, and writes it to ``path``. `overtake.experiments` and
`overtake.train` call them as they produce results, and
`overtake.experiments.figures` redraws the figures from the saved JSON without
rerunning anything.

Importing the module selects matplotlib's non-interactive Agg backend and sets
the house style in ``matplotlib.pyplot.rcParams`` for the whole process.

Colours follow a colour-vision-deficiency-checked palette: up to three
categorical series (blue, orange, green) and a single-hue blue ramp for ordered
values such as battery charge. Every multi-series figure has a legend.
"""
import matplotlib

# Figures are only written to files, so use the file-only backend before pyplot loads a GUI one.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .features import DEFAULTS

# ``matplotlib.ticker`` and ``matplotlib.transforms`` are used below as attributes of ``matplotlib``;
# importing pyplot has already loaded them.
SURFACE = "#fcfcfb"         # figure and axes background
INK = "#0b0b0b"             # titles and primary text
INK_SECONDARY = "#52514e"   # axis labels, legend text, point labels
MUTED = "#898781"           # tick labels, dashed reference lines, notes
GRID = "#e1e0d9"            # grid lines
AXIS = "#c3c2b7"            # axis spines, limit lines, diagram box edges
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                  # categorical slots 1-3
BATTERY_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, low -> high charge

# House style for every figure: light surface, bold left-aligned titles, no top or right spines, a light
# grid and frameless legends.
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
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
    "font.size": 12,
    "lines.linewidth": 2,
})

# Point markers with a background-coloured rim, so neighbouring and overlapping points stay distinct.
MARKER = {"marker": "o", "markersize": 5, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def _percent_axis(ax, axis="y"):
    """Fix the ``axis`` ("x" or "y") of ``ax`` to 0-1 and label its ticks as percentages (0.5 -> "50%")."""
    getattr(ax, f"set_{axis}lim")(0, 1)
    getattr(ax, f"{axis}axis").set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def _log_ticks(ax, values, axis="x"):
    """Label the log ``axis`` ("x" or "y") at exactly ``values`` in plain notation, without minor labels."""
    target = getattr(ax, f"{axis}axis")
    target.set_major_locator(matplotlib.ticker.FixedLocator(values))
    target.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in values]))
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _label_points(ax, xs, ys, labels, marker_pt=4):
    """Label scatter points without labels overlapping each other, the points or the axes edge.

    Greedy placement: the most crowded points are labelled first, while they
    still have free space. Each label tries ten positions around its point (to
    the right, above, below or left, then further above or below) and takes the
    first that stays inside the axes and clears every point, every label placed
    so far and any text already in the axes. If none is free it takes the first
    position (right, slightly above) anyway. Boxes are measured in display
    pixels, so call this after the layout is final (``fig.tight_layout()`` and
    ``fig.canvas.draw()``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xs, ys : sequence of float
        Point positions in data coordinates.
    labels : sequence of str
        One label per point.
    marker_pt : float, optional
        Half-width, in points, of the square kept clear around each point
        (about the marker radius).
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    radius = marker_pt * fig.dpi / 72  # points -> display pixels (72 points per inch)
    points = ax.transData.transform(np.column_stack([xs, ys]))
    # Boxes no label may overlap: every marker, then any text already in the axes (e.g. a statistics note).
    taken = [matplotlib.transforms.Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)
             for px, py in points]
    taken += [t.get_window_extent(renderer) for t in ax.texts]
    # Crowding: points within 60 pixels, the point itself included (threshold: estimate).
    near = [(np.hypot(*(points - pt).T) < 60).sum() for pt in points]
    # (offset in points, horizontal alignment, vertical alignment), in order of preference.
    candidates = [((5, 2), "left", "bottom"), ((5, -2), "left", "top"), ((0, 6), "center", "bottom"),
                  ((0, -6), "center", "top"), ((-5, 2), "right", "bottom"), ((-5, -2), "right", "top"),
                  ((6, 0), "left", "center"), ((-6, 0), "right", "center"),
                  ((0, 14), "center", "bottom"), ((0, -14), "center", "top")]

    def place(i, offset, ha, va):
        """Annotate point ``i`` at ``offset`` points; return the text and its box padded by 1 pixel."""
        text = ax.annotate(labels[i], (xs[i], ys[i]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                           color=INK_SECONDARY, fontsize=10)
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
    """Tighten the layout, write ``fig`` to ``path`` at 160 dpi and close it to free its memory."""
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_curve(history, path):
    """Training and validation loss per epoch.

    Parameters
    ----------
    history : list of dict
        One dict per epoch with ``epoch``, ``train_loss`` and ``val_loss``
        (mean binary cross-entropy, nats), as returned by
        `overtake.train.train_network`.
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

    Predictions are grouped into ``bins`` equal-width probability bins; each
    point is a bin's mean prediction against the share of its attempts that
    were overtakes. Bins with fewer than 20 attempts are left out.

    Parameters
    ----------
    y : numpy.ndarray
        Observed 0/1 outcomes.
    curves : dict of str -> numpy.ndarray
        Model name -> predicted probabilities for the same attempts as ``y``.
        At most three are drawn (one per colour in `SERIES`); any further
        models are ignored.
    path : str or pathlib.Path
        Output PNG.
    bins : int, optional
        Number of equal-width bins on [0, 1].
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    edges = np.linspace(0, 1, bins + 1)
    for (label, p), color in zip(curves.items(), SERIES):
        # Same binning as `overtake.train.expected_calibration_error`; the clip puts p = 1 in the last bin.
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        # Below 20 attempts a bin's observed rate is too noisy to plot (threshold: estimate).
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

    Left: probability against the gap (0.05-2 s) for attacker charges of 25,
    50, 75 and 100 %, with Overtake Mode on only within 1 s as the rules
    allow; the base scenario is marked. Right: probability against the length
    of the straight over 250-1300 m (the range the training data covers for
    straights), with and without the attacker's Boost. Every other input stays
    at ``base``.

    Parameters
    ----------
    predict : callable
        Maps a list of scenario dicts to an array of probabilities (e.g.
        ``OvertakePredictor.predict``).
    path : str or pathlib.Path
        Output PNG.
    exact : callable, optional
        Same signature, giving the simulator's true probability (e.g. a
        wrapper around `overtake.simulator.probability`); drawn dashed for
        comparison.
    base : dict, optional
        Fields that override `overtake.features.DEFAULTS` in the scenario
        varied around.

    Raises
    ------
    ValueError
        If ``predict`` or ``exact`` rejects a scenario;
        ``OvertakePredictor.predict`` does so for an unknown field or an
        out-of-range value in ``base``.
    """
    base = {**DEFAULTS, **(base or {})}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)

    def draw(ax, xs, scenarios, color, label):
        """Plot ``predict`` over ``scenarios`` against ``xs`` as a solid line and, if given, ``exact`` dashed."""
        ax.plot(xs, predict(scenarios), color=color, label=label)
        if exact is not None:
            ax.plot(xs, exact(scenarios), color=color, linewidth=1, linestyle="--")

    gaps = np.linspace(0.05, 2.0, 80)
    for battery, color in zip((25, 50, 75, 100), BATTERY_RAMP):
        # Overtake Mode needs a gap of at most 1 s at the detection point, so it switches off beyond 1 s.
        draw(left, gaps, [{**base, "gap_s": g, "overtake_mode": g <= 1.0, "attacker_battery": battery}
                          for g in gaps], color, f"{battery}% battery")
    p_base = float(np.asarray(predict([base]))[0])
    left.plot([base["gap_s"]], [p_base], marker="o", markersize=8, color=BATTERY_RAMP[-1],
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    left.annotate(f"Default scenario: {p_base:.0%}", (base["gap_s"], p_base), xytext=(10, 0),
                  textcoords="offset points", va="center", color=INK)
    left.axvline(1.0, color=AXIS, linewidth=1)
    # Right of the line, between the legend and the default-scenario label, where no curve reaches.
    left.annotate("Overtake Mode\nlimit (1 s)", (1.0, 0.535), xytext=(6, 0), textcoords="offset points",
                  ha="left", va="center", color=MUTED, fontsize=11)
    left.set_xlabel("Gap to the car ahead (s)")
    left.set_ylabel("P(overtake)")
    left.set_title("Gap and attacker battery charge")
    left.legend(loc="upper right")
    _percent_axis(left)  # also applies to the right panel (shared y)

    # 250-1300 m: the straights the training data covers (`overtake.features.TRAINED_RUN_LENGTH_M`).
    lengths = np.linspace(250, 1300, 80)
    for boost, label, color in ((True, "With Boost", SERIES[0]), (False, "Without Boost", SERIES[1])):
        draw(right, lengths, [{**base, "run_length_m": L, "attacker_boost": boost} for L in lengths], color, label)
    right.set_xlabel("Length of the straight (m)")
    right.set_title(f"Straight length, {base['gap_s']:g} s gap")
    right.legend(loc="upper left")
    if exact is not None:
        right.annotate("Solid: neural network\nDashed: simulator truth", (0.02, 0.62),
                       xycoords="axes fraction", ha="left", va="top", color=MUTED)
    _save(fig, path)


def convergence(results, path):
    """Error against time step for Euler and RK4 on problems with analytical solutions.

    One log-log panel per problem. Dashed grey lines of slope 1 and 4 through
    each method's error at the last (smallest) step show the theoretical
    orders.

    Parameters
    ----------
    results : dict
        `overtake.experiments.solver` output: ``{problem_title: {"dt": [...],
        "euler": [...], "rk4": [...], "euler_order": float, "rk4_order":
        float}}``, with steps in s listed from largest to smallest, final
        position errors in m and the observed orders (least-squares slopes)
        quoted in the legend.
    path : str or pathlib.Path
        Output PNG.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(10, 3.8))
    # atleast_1d: with one problem, plt.subplots returns a single Axes rather than an array.
    for ax, (title, r) in zip(np.atleast_1d(axes), results.items()):
        dt = np.array(r["dt"])
        for method, label, color, order in (("rk4", "RK4", SERIES[0], 4), ("euler", "Euler", SERIES[1], 1)):
            err = np.array(r[method])
            ax.loglog(dt, err, color=color, label=f"{label} (observed order {r[method + '_order']:.2f})", **MARKER)
            # Theoretical error C * dt^order, with C set so the line passes through the smallest step's error.
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

    Left: the largest braking-point deficit error over the test scenarios
    against time step, with the 1 ms acceptance limit. Right: the same error
    against run time, i.e. accuracy against cost. The chosen step is
    highlighted in both.

    Parameters
    ----------
    results : dict
        `overtake.experiments.timestep` output: ``{"dt": [...], "rk4": {...},
        "euler": {...}}``, where each method has ``max_error_s`` (largest
        deficit error against the fine-step reference, s) and ``seconds``
        (run time for all the test scenarios, s); optionally
        ``rk4_linear_crossing`` in the same form (RK4 with a linear instead
        of a Hermite braking-point crossing; drawn dashed, accuracy only).
    path : str or pathlib.Path
        Output PNG.
    chosen_dt : float
        Step used by the simulator (`overtake.simulator.TIME_STEP_S`),
        highlighted; must be one of ``results["dt"]``.

    Raises
    ------
    ValueError
        If ``chosen_dt`` is not in ``results["dt"]``.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    dt = np.array(results["dt"])
    for method, label, color in (("rk4", "RK4", SERIES[0]), ("euler", "Euler", SERIES[1])):
        err, sec = np.array(results[method]["max_error_s"]), np.array(results[method]["seconds"])
        left.loglog(dt, err, color=color, label=label, **MARKER)
        right.loglog(sec, err, color=color, label=label, **MARKER)
    if "rk4_linear_crossing" in results:
        left.loglog(dt, results["rk4_linear_crossing"]["max_error_s"], color=SERIES[0], linewidth=1,
                    linestyle="--", label="RK4, linear crossing")
    i = list(dt).index(chosen_dt)  # exact match: the chosen step must be one of the studied steps
    # Label below-right of the point on the left panel (the curves rise to the right there), left of it on the right.
    for ax, x, offset, ha, va in ((left, chosen_dt, (8, -10), "left", "top"),
                                  (right, results["rk4"]["seconds"][i], (-10, 0), "right", "center")):
        ax.plot([x], [results["rk4"]["max_error_s"][i]], marker="o", markersize=10, color=SERIES[0],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"chosen: dt = {chosen_dt} s", (x, results["rk4"]["max_error_s"][i]), xytext=offset,
                    textcoords="offset points", ha=ha, va=va, color=INK)
        ax.axhline(1e-3, color=AXIS, linewidth=1)  # 1 ms limit on the deficit error
    left.annotate("1 ms acceptance limit", (dt.min(), 1e-3), xytext=(0, 4), textcoords="offset points",
                  ha="left", color=MUTED)
    _log_ticks(left, sorted(dt))
    left.set_xlabel("Time step (s)")
    left.set_ylabel("Max deficit error (s)")
    left.set_title("Accuracy of the braking-point deficit")
    left.legend(loc="upper left")
    right.set_xlabel("Run time for the test scenarios (s)")
    right.set_title("Cost against accuracy")
    right.legend(loc="upper right")
    _save(fig, path)


def stability(results, path):
    """Energy-budget overshoot against time step, Euler and RK4, with the predicted stability limits.

    One panel per battery deployment. Near an empty battery the deployed
    power ramps down as a fast decay with rate lambda, which an explicit
    method follows stably only for dt below its real-axis stability bound
    divided by lambda (2 / lambda for Euler, 2.785 / lambda for RK4). These
    predicted limits are drawn dashed in each method's colour, the chosen step
    in grey.

    Parameters
    ----------
    results : dict
        `overtake.experiments.stability` output: ``dt`` (steps, s),
        ``chosen_dt`` (s) and ``deployments``, a dict of panel title ->
        ``{"decay_rate_per_s": float, "predicted_dt_limit_s": {"euler":
        float, "rk4": float}, "euler": {"max_energy_overshoot_j": [...]},
        "rk4": {...}}``. Needs at least two deployments.
    path : str or pathlib.Path
        Output PNG.

    Raises
    ------
    TypeError
        With a single deployment, since ``plt.subplots`` then returns one
        Axes rather than an array.
    """
    deployments = results["deployments"]
    fig, axes = plt.subplots(1, len(deployments), figsize=(10, 3.6), sharey=True)
    dt = np.array(results["dt"])
    for ax, (title, row) in zip(axes, deployments.items()):
        for method, label, color in (("rk4", "RK4", SERIES[0]), ("euler", "Euler", SERIES[1])):
            # J -> kJ
            ax.plot(dt, np.array(row[method]["max_energy_overshoot_j"]) / 1e3, color=color, label=label, **MARKER)
            limit = row["predicted_dt_limit_s"][method]
            ax.axvline(limit, color=color, linewidth=1, linestyle="--")
            # x in data units, y as a fraction of the axes height: the label hangs from the top edge.
            ax.annotate(f"{label} limit\n{limit:.2f} s", (limit, 1.0), xycoords=ax.get_xaxis_transform(),
                        xytext=(3, -3), textcoords="offset points", va="top", color=color, fontsize=10)
        ax.axvline(results["chosen_dt"], color=AXIS, linewidth=1)
        ax.annotate(f"chosen\n{results['chosen_dt']} s", (results["chosen_dt"], 0.35),
                    xycoords=ax.get_xaxis_transform(), xytext=(4, 0), textcoords="offset points",
                    ha="left", color=MUTED, fontsize=10)
        ax.set_title(f"{title}: decay rate {row['decay_rate_per_s']:.1f} 1/s")
        ax.set_xlabel("Time step (s)")
    # One legend, in the empty space between the chosen step and the limits.
    axes[-1].legend(loc="upper left", bbox_to_anchor=(0.2, 0.97))
    axes[0].set_ylabel("Energy used beyond the battery (kJ)")
    _save(fig, path)


def quadrature(results, path):
    """Quadrature rule convergence and Monte Carlo error.

    Left: the largest error in P over the test scenarios against the number
    of integrand evaluations, per rule. Right: the RMS error of the Monte Carlo
    estimate against the number of replays, beside the binomial theory
    sqrt(p(1-p)/N).

    Parameters
    ----------
    results : dict
        `overtake.experiments.quadrature` output: ``{"rules": {rule:
        {"evaluations": [...], "max_error": [...]}}, "monte_carlo": {"runs":
        [...], "rms_error": [...], "theory": [...]}}``. At most three rules
        are drawn (one per colour in `SERIES`).
    path : str or pathlib.Path
        Output PNG.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for (rule, r), color in zip(results["rules"].items(), SERIES):
        err = np.maximum(r["max_error"], 1e-17)  # floor so an error of exactly 0 can still go on a log axis
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
        One per network size (`overtake.experiments.selection`), each with
        ``label`` (hidden widths, e.g. "64-64"), ``parameters`` (trainable
        count) and ``val_mae_vs_true`` (mean |p - p_true| on the validation
        set, a probability).
    data_sizes : list of dict
        One per training-set size, each with ``train_size`` (attempts) and
        ``val_mae_vs_true``.
    path : str or pathlib.Path
        Output PNG.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    params = [a["parameters"] for a in architectures]
    mae = [100 * a["val_mae_vs_true"] for a in architectures]  # probability -> percentage points
    left.semilogx(params, mae, color=SERIES[0], **MARKER)
    for a, x, y in zip(architectures, params, mae):
        left.annotate(a["label"], (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                      color=INK_SECONDARY)
    left.set_xlabel("Trainable parameters")
    left.set_ylabel("Validation error vs truth (points)")
    left.set_title("Network size")
    left.set_ylim(0, max(mae) * 1.3)  # headroom for the labels above the points

    sizes = [d["train_size"] for d in data_sizes]
    right.semilogx(sizes, [100 * d["val_mae_vs_true"] for d in data_sizes], color=SERIES[0], **MARKER)
    _log_ticks(right, sizes)
    right.set_xlabel("Training attempts")
    right.set_title("Training-set size")
    right.set_ylim(0, None)
    _save(fig, path)


def method_comparison(regions, path):
    """Mean error against the true probability per operating region and model, as grouped bars.

    Parameters
    ----------
    regions : dict
        ``{model: {region: mae}}`` with the same regions, in the same order,
        for every model; ``mae`` is the mean |p - p_true| (a probability). At
        most three models are drawn (one per colour in `SERIES`).
    path : str or pathlib.Path
        Output PNG.
    """
    models = list(regions)
    names = list(regions[models[0]])
    x = np.arange(len(names))
    width = 0.8 / len(models)  # a region's group of bars spans 0.8 of the spacing between regions
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (model, color) in enumerate(zip(models, SERIES)):
        values = [100 * regions[model][r] for r in names]
        # Offset each model's bar from the region's tick so the group is centred; 0.92 leaves a gap between bars.
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width * 0.92, color=color, label=model)
    # Every underscore -> line break: "tyres_suit_surface" -> "tyres\nsuit\nsurface".
    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylabel("Mean error vs truth (points)")
    ax.set_title("Error by operating region (test set)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


# Display names of the `overtake.features` inputs, for the sensitivity figure's row labels.
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

    Each input is one row: a bar from the network's probability at the low
    setting to the one at the high setting, the simulator's true
    probabilities as orange ticks, and the settings' labels outside both. The
    first entry of ``rows`` is drawn at the top.

    Parameters
    ----------
    rows : list of dict
        From `overtake.experiments.sensitivity`, each with ``input`` (column
        name, shown through `READABLE`), ``low`` and ``high`` (labels of the
        two settings, e.g. "0.3 s"), ``network`` and ``exact`` (pairs of
        probabilities at the low and high settings), sorted most important
        first.
    path : str or pathlib.Path
        Output PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))  # 0.42 in per row plus title and axis
    # Reversed: row 0 is drawn at the bottom, so the most important input ends up at the top.
    for i, row in enumerate(reversed(rows)):
        lo, hi = row["network"]
        ax.plot([lo, hi], [i, i], color=SERIES[0], linewidth=3, solid_capstyle="round",
                label="Neural network" if i == 0 else None)
        ax.plot(row["exact"], [i, i], linestyle="none", marker="|", markersize=12, markeredgewidth=2,
                color=SERIES[1], label="Simulator truth" if i == 0 else None)
        # Label each setting on the outside of the bar and its truth marker. Sorting by the network's
        # probability puts whichever setting gives the lower probability on the left.
        ends = sorted([(lo, row["exact"][0], row["low"]), (hi, row["exact"][1], row["high"])])
        (l_net, l_true, l_text), (r_net, r_true, r_text) = ends
        ax.annotate(l_text, (min(l_net, l_true), i), xytext=(-7, 0), textcoords="offset points", ha="right",
                    va="center", color=MUTED, fontsize=10)
        ax.annotate(r_text, (max(r_net, r_true), i), xytext=(7, 0), textcoords="offset points", ha="left",
                    va="center", color=MUTED, fontsize=10)
    ax.set_yticks(range(len(rows)), [READABLE.get(r["input"], r["input"]) for r in reversed(rows)])
    _percent_axis(ax, "x")
    ax.set_xlim(-0.12, 1.12)  # room beyond 0 % and 100 % for the setting labels
    ax.set_xlabel("P(overtake)")
    ax.set_title("Sensitivity around the default scenario")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def constants(results, path):
    """Change in the exact probability when each estimated constant moves down or up by a fixed fraction.

    Parameters
    ----------
    results : dict
        `overtake.experiments.constants` output: ``change`` (relative change,
        0.1 = 10 %) and ``rows``, each with ``constant`` (display name) and
        ``mean_abs_change`` (mean absolute change in P over the random
        scenarios with the constant lowered, then raised; probabilities),
        sorted most important first.
    path : str or pathlib.Path
        Output PNG.
    """
    rows = results["rows"]
    pct = round(100 * results["change"])
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]  # first (most important) row at the top
    low = np.array([r["mean_abs_change"][0] for r in rows]) * 100  # probability -> percentage points
    high = np.array([r["mean_abs_change"][1] for r in rows]) * 100
    # Two bars per row, 0.36 high and centred 0.19 above and below it, so they almost touch.
    ax.barh(y + 0.19, low, height=0.36, color=SERIES[0], label=f"{pct} % lower")
    ax.barh(y - 0.19, high, height=0.36, color=SERIES[1], label=f"{pct} % higher")
    ax.set_yticks(y, [r["constant"] for r in rows])
    ax.set_xlabel("Mean change in P(overtake), percentage points")
    ax.set_title(f"Effect of each estimated constant changed by {pct} %")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def rules(results, path):
    """Exact probability against the Overtake Mode energy and taper start, by attacker charge.

    Each panel varies one rule with the other held at its 2026 value; a
    vertical line marks the 2026 value of the rule varied.

    Parameters
    ----------
    results : dict
        `overtake.experiments.rules` output: ``batteries`` (attacker charges,
        %), ``energy`` (``mj`` and ``p``), ``taper`` (``start_kph`` and
        ``p``), where ``p`` maps each charge, as a string, to the default
        scenario's exact P at each setting, and ``rule_2026`` (``energy_mj``,
        ``taper_start_kph``). At most four charges are drawn (one per colour
        in `BATTERY_RAMP`).
    path : str or pathlib.Path
        Output PNG.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ramp = BATTERY_RAMP[-len(results["batteries"]):]  # the darkest colours, so the highest charge is darkest
    for battery, color in zip(results["batteries"], ramp):
        left.plot(results["energy"]["mj"], results["energy"]["p"][str(battery)], color=color,
                  label=f"{battery}% battery", **MARKER)
        right.plot(results["taper"]["start_kph"], results["taper"]["p"][str(battery)], color=color,
                   label=f"{battery}% battery", **MARKER)
    rule = results["rule_2026"]
    for ax, x, text in ((left, rule["energy_mj"], f"2026: {rule['energy_mj']:g} MJ"),
                        (right, rule["taper_start_kph"], f"2026: {rule['taper_start_kph']:g} km/h")):
        ax.axvline(x, color=AXIS, linewidth=1)
        ax.annotate(text, (x, 0.97), xycoords=ax.get_xaxis_transform(), xytext=(4, 0),
                    textcoords="offset points", va="top", color=MUTED)
    left.set_xlabel("Overtake Mode extra energy (MJ)")
    left.set_ylabel("P(overtake), default scenario")
    # The titles state the 2026 value held fixed in the other panel (`overtake.physics.TAPER_OVERTAKE_MODE_KPH`,
    # `overtake.physics.OVERTAKE_MODE_EXTRA_J`); they are typed in, not read from ``results``.
    left.set_title("Extra energy (taper start 337 km/h)")
    left.legend(loc="upper left")
    right.set_xlabel("Speed where electrical power starts to taper (km/h)")
    right.set_title("Taper start (extra energy 0.5 MJ)")
    _percent_axis(left)  # also applies to the right panel (shared y)
    _save(fig, path)


def speed_trace(traces, path):
    """Speed and battery energy of both cars along the straight.

    Top row: speed; bottom row: battery energy used; one column per trace.
    Both cars are drawn against the defender's distance from the corner exit,
    so at any x the two curves show the cars at the same instant.

    Parameters
    ----------
    traces : dict of str -> pandas.DataFrame
        Panel title -> output of `overtake.simulator.speed_trace` (two in the
        report; the figure is 5 in wide per trace).
    path : str or pathlib.Path
        Output PNG.
    """
    # squeeze=False keeps ``axes`` 2-D even with a single trace.
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

    Left: the longest full-throttle stretch of one car in one race
    (`overtake.realdata.TELEMETRY`) against the car model run from the same
    entry speed for the same time with each deployment. Right: per
    deployment, the spread of the RMS speed error over every full-throttle
    stretch, labelled with how often that deployment fitted best.

    Parameters
    ----------
    result : dict
        ``telemetry`` part of `overtake.realdata.analyse`: ``traces``
        (``real`` and one per deployment, each with ``x`` in m and ``kph``),
        ``rms_kph`` (per deployment, one RMS error per stretch, km/h),
        ``best_fit`` (per deployment, the number of stretches it fits best),
        ``runs`` (number of stretches) and ``car`` (with ``driver_number``).
        At most three deployments are drawn (one per colour in `SERIES`).
    path : str or pathlib.Path
        Output PNG.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    traces = result["traces"]
    models = [k for k in traces if k != "real"]
    for name, color in zip(models, SERIES):
        left.plot(traces[name]["x"], traces[name]["kph"], color=color, label=f"Model: {name}")
    left.plot(traces["real"]["x"], traces["real"]["kph"], color=INK, linewidth=2.5, label="Real (OpenF1)")
    left.set_xlabel("Distance along the straight (m)")
    left.set_ylabel("Speed (km/h)")
    # "Monza" is typed in: it is the race `overtake.realdata.TELEMETRY` selects.
    left.set_title(f"Longest straight, car {result['car']['driver_number']}, Monza 2026")
    left.legend(loc="lower right")

    data = [result["rms_kph"][m] for m in models]
    box = right.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                        medianprops={"color": INK, "linewidth": 1.5}, flierprops={"markersize": 3})
    for patch, color in zip(box["boxes"], SERIES):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Horizontal boxes sit at y = 1, 2, ... in the order of ``models``.
    right.set_yticks(range(1, len(models) + 1),
                     [f"{m}\nbest fit on {result['best_fit'][m]} of {result['runs']}" for m in models])
    right.set_xlabel("RMS speed error per straight (km/h)")
    right.set_title("All full-throttle straights")
    right.grid(axis="y", visible=False)
    _save(fig, path)


def real_gap(result, path):
    """Pass rate against the gap to the car ahead: real races and the simulator.

    Left, the shape only: the real per-lap pass rate and the simulator's mean
    per-attempt probability measure different things, so each is divided by
    its own value in the 0.25-0.5 s bin. Right: the real per-lap pass rate
    itself, with 95 % Wilson intervals.

    Parameters
    ----------
    result : dict
        ``gap`` part of `overtake.realdata.analyse`: ``real`` (one dict per
        gap bin with ``low`` and ``high`` in s, ``rate``, ``ci`` (95 %
        interval of the rate) and ``relative``), ``simulated`` (per bin ``n``
        and ``relative``) and ``driver_laps``. The second bin must be the
        0.25-0.5 s reference bin.
    path : str or pathlib.Path
        Output PNG.
    """
    real, sim = result["real"], result["simulated"]
    mids = [(r["low"] + r["high"]) / 2 for r in real]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9))
    ref = real[1]["rate"]  # the 0.25-0.5 s bin that ``relative`` is measured against
    # Error bars: the rate's interval divided by the same reference; the reference's own uncertainty is ignored.
    lo = [r["relative"] - r["ci"][0] / ref for r in real]
    hi = [r["ci"][1] / ref - r["relative"] for r in real]
    left.errorbar(mids, [r["relative"] for r in real], yerr=[lo, hi], color=INK, label="Real 2026 races (per lap)",
                  capsize=3, **MARKER)
    # Bins beyond the simulated gaps hold no attempts (n = 0) and get no simulator point.
    sim_mids = [m for m, q in zip(mids, sim) if q["n"]]
    left.plot(sim_mids, [q["relative"] for q in sim if q["n"]], color=SERIES[1],
              label="Simulator (per attempt)", **MARKER)
    left.axvline(1.0, color=AXIS, linewidth=1)
    left.annotate("Overtake Mode limit", (1.0, 0.5), xycoords=left.get_xaxis_transform(), xytext=(4, 0),
                  textcoords="offset points", rotation=90, va="center", color=MUTED, fontsize=10)
    left.set_xlabel("Gap to the car ahead (s)")
    left.set_ylabel("Pass rate relative to 0.25-0.5 s")
    left.set_title("Shape: how fast the chance falls with gap")
    left.set_ylim(0, None)
    left.legend(loc="upper right")

    rates = [100 * r["rate"] for r in real]  # share -> %
    err = [[100 * (r["rate"] - r["ci"][0]) for r in real], [100 * (r["ci"][1] - r["rate"]) for r in real]]
    right.errorbar(mids, rates, yerr=err, color=INK, capsize=3, **MARKER)
    right.axvline(1.0, color=AXIS, linewidth=1)
    right.set_xlabel("Gap at the start of the lap (s)")
    right.set_ylabel("Laps with an on-track pass (%)")
    right.set_title(f"Real 2026 races ({result['driver_laps']:,} driver-laps)")
    right.set_ylim(0, None)
    _save(fig, path)


def real_circuits(result, path):
    """Real on-track passes per lap against the simulator's circuit passability.

    One labelled point per race whose circuit has a passability in
    `overtake.simulator.PASSABILITY`, with the Spearman rank correlation
    computed by `overtake.realdata.analyse`.

    Parameters
    ----------
    result : dict
        `overtake.realdata.analyse` output. Uses ``races`` (each with
        ``race``, ``on_track_per_lap`` and, for modelled circuits,
        ``passability``) and ``circuits`` (``spearman_rho``, ``p_value``,
        ``n``).
    path : str or pathlib.Path
        Output PNG.
    """
    rows = [r for r in result["races"] if r.get("passability") is not None]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot([r["passability"] for r in rows], [r["on_track_per_lap"] for r in rows], linestyle="none",
            color=SERIES[0], **{**MARKER, "markersize": 7})
    c = result["circuits"]
    ax.annotate(f"Spearman rank correlation {c['spearman_rho']:.2f} (p = {c['p_value']:.3f}, {c['n']} circuits)",
                (0.02, 0.96), xycoords="axes fraction", va="top", color=INK)
    ax.set_xlabel("Simulator passability (estimate, log-odds)")
    ax.set_ylabel("Real on-track passes per race lap")
    ax.set_title("Circuits: model assumption against 2026 races")
    # `_label_points` measures boxes in display pixels, so fix the layout and render once first.
    fig.tight_layout()
    fig.canvas.draw()
    _label_points(ax, [r["passability"] for r in rows], [r["on_track_per_lap"] for r in rows],
                  [r["race"].replace("-Francorchamps", "") for r in rows])  # "Spa-Francorchamps" -> "Spa"
    _save(fig, path)


def real_speed_traps(rows, path):
    """Speed-trap readings per race against the 2026 MGU-K speed limits.

    One row per race, the fastest circuit (by median) at the top: a grey line
    from the median lap to the fastest reading, with markers for the median,
    the 99th percentile and the maximum. Vertical lines mark 337 km/h, where
    full power ends with Overtake Mode, and 355 km/h, where electric power
    reaches zero.

    Parameters
    ----------
    rows : list of dict
        ``speed_traps`` part of `overtake.realdata.analyse`, each with
        ``race`` and the ``median``, ``p99`` and ``max`` over its laps of the
        highest timing-point speed per lap (km/h).
    path : str or pathlib.Path
        Output PNG.
    """
    rows = sorted(rows, key=lambda r: r["median"])  # slowest at y = 0 (bottom)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.hlines(y, [r["median"] for r in rows], [r["max"] for r in rows], color=GRID, linewidth=3)
    ax.plot([r["median"] for r in rows], y, linestyle="none", color=SERIES[0], label="Median lap", **MARKER)
    ax.plot([r["p99"] for r in rows], y, linestyle="none", color=SERIES[1], label="99th percentile", **MARKER)
    ax.plot([r["max"] for r in rows], y, linestyle="none", marker="|", markersize=10, markeredgewidth=2,
            color=INK, label="Fastest reading")
    # The two speeds of the Overtake Mode taper (`overtake.physics.TAPER_OVERTAKE_MODE_KPH`, typed in),
    # labelled just above the plot so the labels never sit on data.
    for x, label, ha in ((337, "337: full power\nends (Overtake Mode)", "right"),
                         (355, "355: electric\npower zero", "center")):
        ax.axvline(x, color=AXIS, linewidth=1)
        ax.annotate(label, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha=ha, va="bottom", color=MUTED, fontsize=10)
    ax.set_yticks(y, [r["race"] for r in rows])
    ax.set_xlabel("Highest speed per lap from the timing points (km/h)")
    ax.set_title("Speed traps, every 2026 race lap", pad=34)
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def workflow_diagram(path):
    """Boxes-and-arrows overview of the computational workflow.

    Top row: the path from a scenario through the simulator to the network's
    training data. Bottom row: the exact probability, the network's
    predictions and the checks on each part (verification, validation, real
    races). Colours: blue for the simulator, orange for the network, green
    for verification and validation. Written at 200 dpi and cropped to the
    boxes.

    Parameters
    ----------
    path : str or pathlib.Path
        Output PNG.
    """
    # Sized for the full text width of an A4 page (about 6.9 in), so the text prints at about 7-8 pt.
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    # Boxes are laid out in data units on a 10 x 3.6 canvas.
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    tint = {"sim": "#e6effa", "nn": "#fde9df", "val": "#e3f5ee"}  # light tints of the three `SERIES` colours

    def box(x, y, w, h, title, body, kind):
        """Draw a rounded box with a bold title and a smaller description; return its extent ``(x, y, w, h)``."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=tint[kind], edgecolor=AXIS, linewidth=0.8))
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.4, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=7.0, color=INK_SECONDARY,
                linespacing=1.3)
        return (x, y, w, h)

    def arrow(a, b, label=None, dy=0.0):
        """Draw an arrow from point ``a`` to point ``b``, optionally labelled at its midpoint (raised by ``dy``)."""
        (x0, y0), (x1, y1) = a, b
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     color=INK_SECONDARY, linewidth=0.9))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08 + dy, label, ha="center", va="bottom", fontsize=7.2,
                    color=INK_SECONDARY)

    # Top row: the simulator and the network; bottom row: the exact probability, the prediction and the checks.
    # ``top`` and ``bottom`` are the rows' lower edges and ``h`` the box height, in data units.
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
        """Midpoint of the right, left, top or bottom edge ("r", "l", "t", "b") of box ``b``."""
        x, y, w, h = b
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]
    arrow(mid(scen, "r"), mid(ode, "l"))
    arrow(mid(ode, "r"), mid(outc, "l"))
    arrow(mid(outc, "r"), mid(lab, "l"))
    arrow(mid(lab, "r"), mid(net, "l"))
    arrow(mid(outc, "b"), mid(exact, "t"))
    arrow(mid(exact, "r"), mid(valid, "l"))
    arrow(mid(net, "b"), mid(pred, "t"))
    arrow((8.2, top), (7.9, bottom + h))           # network -> validation (corner to corner)
    arrow(mid(verif, "t"), mid(ode, "b"))           # verification checks the ODE solver
    arrow((1.7, bottom + h), (2.0, top))            # real races check the physics
    # Not `_save`: bbox_inches="tight" crops the empty margins around the hidden axes.
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
