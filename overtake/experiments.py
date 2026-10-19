"""Experiments behind the project documentation.

    python -m overtake.experiments all
    python -m overtake.experiments solver timestep      # a subset

Each experiment writes ``reports/results/<name>.json`` and a figure in
``reports/figures/``.

| Name | What it does |
|---|---|
| `solver` | Euler and RK4 against analytical solutions; observed order |
| `timestep` | Error and cost of the full two-car model against time step |
| `quadrature` | Quadrature rules and Monte Carlo against a reference |
| `selection` | Network size and training-set size, on the validation set |
| `ablation` | Physics-informed inputs and five-network ensembles, five seeds |
| `comparison` | Network vs logistic regression vs gradient boosting vs direct simulation, on the test set |
| `performance` | Profiling, vectorisation, chunk size, FLOPs and memory |
| `sensitivity` | One-at-a-time sensitivity, network vs simulator truth |

``comparison`` and ``sensitivity`` use the trained network in ``models/``,
so run ``python -m overtake.train`` first.
"""
import cProfile
import functools
import io
import json
import pickle
import pstats
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from . import baselines, plots, simulator
from .features import CATEGORICAL, DEFAULTS, LABEL, VOCABS
from .model import OvertakeNet
from .physics import CarParameters, CarState, acceleration, integrate
from .predict import DEFAULT_MODEL, OvertakePredictor
from .train import REPORTS, metrics, split, train_network

FIGURES, RESULTS = REPORTS / "figures", REPORTS / "results"
SEED = 0


def _write(name, result):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def _timed(func, repeats=3):
    """Best wall-clock time of ``repeats`` calls, and the last result."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return best, out


@functools.lru_cache(maxsize=None)
def dataset(n=300_000, seed=SEED):
    """Simulated attempts, cached so experiments share one dataset."""
    return simulator.generate_dataset(n, seed=seed)


def nn_flops(hidden, n_inputs=None):
    """Floating-point operations for one network prediction.

    Counts a multiply-add as 2 FLOPs, a bias add as 1 and a SiLU activation as
    about 5 (exponential, add, divide, multiply). Embedding look-ups are memory
    reads, not arithmetic.

    Parameters
    ----------
    hidden : tuple of int
    n_inputs : int, optional
        Width of the first layer's input (embeddings + numeric inputs).

    Returns
    -------
    int
    """
    if n_inputs is None:
        n_inputs = OvertakeNet(hidden).mlp[0].in_features
    widths = [n_inputs, *hidden, 1]
    linear = sum(2 * a * b + b for a, b in zip(widths[:-1], widths[1:]))
    return linear + 5 * sum(hidden)


# Hand count of the arithmetic in physics.two_car_rhs: about 35 FLOPs per car
# in `acceleration` plus about 20 for the slipstream (an exponential counted
# as ~10). RK4 adds 4 right-hand-side evaluations and ~14 vector operations
# per state variable per step.
FLOPS_PER_RHS = 2 * 35 + 20
FLOPS_PER_RK4_STEP = 4 * FLOPS_PER_RHS + 6 * 14
FLOPS_QUADRATURE = 16 * 30 + 40  # 16 Gauss nodes (density + logistic) and the closed-form tail


# --- 1. Solver verification -------------------------------------------------

def _single_car(car):
    state = CarState(mu=np.array([1e9]), cda_m2=np.array([car.cda_m2]),
                     deploy_share=np.array([0.0]), energy_j=np.array([0.0]))

    def f(y):
        accel, ers = acceleration(y[1:2], y[2:3], state, state.cda_m2, car)
        return np.array([y[1], accel[0], ers[0]])
    return f


def solver():
    """Verify Euler and RK4 on two special cases of the car model with exact solutions.

    * Constant power ``P``, no resistance: ``v^2 = v0^2 + 2 P t / m`` and
      ``x = m / (3P) (v^3 - v0^3)``.
    * Drag only: ``v = v0 / (1 + k v0 t / m)`` and ``x = (m / k) ln(1 + k v0 t / m)``.

    The observed order of convergence is the slope of log(error) against
    log(dt) over the smallest steps.
    """
    m, p, k, v0, t_end = 800.0, 500e3, 0.6, 30.0, 12.8
    power_only = replace(CarParameters(), mass_kg=m, ice_power_w=p, cda_m2=0.0, cla_m2=0.0,
                         rolling_coeff=0.0, drivetrain_efficiency=1.0)
    drag_only = replace(power_only, ice_power_w=0.0, cda_m2=2 * k / 1.2, air_density=1.2)
    problems = {
        "Constant power, no drag": (power_only, m / (3 * p) * ((v0**2 + 2 * p * t_end / m) ** 1.5 - v0**3)),
        "Drag only, no power": (drag_only, m / k * np.log(1 + k * v0 * t_end / m)),
    }
    dts = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    out = {}
    for title, (car, x_exact) in problems.items():
        r = {"dt": dts, "exact_position_m": x_exact}
        for method in ("euler", "rk4"):
            r[method] = [abs(integrate(_single_car(car), [0.0, v0, 0.0], dt, int(round(t_end / dt)), method)[0]
                             - x_exact) for dt in dts]
            tail = slice(-4, None) if method == "euler" else slice(1, 5)  # asymptotic, above round-off
            r[method + "_order"] = float(np.polyfit(np.log(dts[tail]), np.log(r[method][tail]), 1)[0])
        out[title] = r
    plots.convergence(out, FIGURES / "solver_convergence.png")
    return _write("solver", out)


# --- 2. Time-step study ------------------------------------------------------

def timestep(n=2000, dts=(0.4, 0.2, 0.1, 0.05, 0.025, 0.0125), reference_dt=0.001):
    """Deficit error of the full model against a fine-step RK4 reference.

    The full model has kinks (traction limit, MGU-K taper, energy running
    out), so the observed order is lower than the smooth-case order; this
    study chooses the step that keeps the error well below the 1 ms
    acceptance limit (0.5 % of the 0.2 s execution noise).
    """
    df = simulator.sample_scenarios(n, np.random.default_rng(7))
    reference = simulator.closing_run(df, dt=reference_dt)[0].deficit_s
    out = {"dt": list(dts), "reference_dt": reference_dt, "scenarios": n}
    for method in ("rk4", "euler"):
        rows = {"max_error_s": [], "mean_error_s": [], "seconds": []}
        for dt in dts:
            seconds, (run, _, _) = _timed(lambda: simulator.closing_run(df, dt=dt, method=method), repeats=2)
            err = np.abs(run.deficit_s - reference)
            rows["max_error_s"].append(float(err.max()))
            rows["mean_error_s"].append(float(err.mean()))
            rows["seconds"].append(seconds)
        fine = slice(1, None)
        rows["observed_order"] = float(np.polyfit(np.log(dts[fine]), np.log(rows["mean_error_s"][fine]), 1)[0])
        out[method] = rows
    plots.timestep(out, FIGURES / "timestep_study.png", simulator.TIME_STEP_S)
    return _write("timestep", out)


# --- 3. Quadrature and Monte Carlo ------------------------------------------

def quadrature(n=2000, mc_scenarios=200):
    """Accuracy of the quadrature rules and of Monte Carlo replays.

    Reference: 200-node Gauss-Legendre (agrees with SciPy's adaptive ``quad``
    to 1e-10, see ``tests/test_simulator.py``).
    """
    df = simulator.sample_scenarios(n, np.random.default_rng(11))
    params = simulator.outcome_parameters(df, *simulator.closing_run(df))
    reference = simulator.overtake_integral(*params, rule="gauss", n=200)
    rules = {
        "Gauss-Legendre": ("gauss", [2, 4, 8, 12, 16, 24, 32], lambda k: k),
        "Simpson": ("simpson", [2, 4, 8, 16, 32, 64, 128, 256], lambda k: k + 1),
        "Trapezoid": ("trapezoid", [2, 4, 8, 16, 32, 64, 128, 256], lambda k: k + 1),
    }
    out = {"rules": {}}
    for title, (rule, sizes, evaluations) in rules.items():
        out["rules"][title] = {
            "n": sizes,
            "evaluations": [evaluations(k) for k in sizes],
            "max_error": [float(np.abs(simulator.overtake_integral(*params, rule=rule, n=k) - reference).max())
                          for k in sizes],
        }

    sub = df.iloc[:mc_scenarios]
    exact = reference[:mc_scenarios]
    runs = [100, 400, 1600, 6400, 25600]
    out["monte_carlo"] = {
        "runs": runs,
        "rms_error": [float(np.sqrt(np.mean((simulator.monte_carlo_probability(sub, r, seed=3) - exact) ** 2)))
                      for r in runs],
        "theory": [float(np.sqrt(np.mean(exact * (1 - exact)) / r)) for r in runs],
    }
    seconds_gauss, _ = _timed(lambda: simulator.overtake_integral(*params))
    seconds_mc, _ = _timed(lambda: simulator.draw_outcomes(
        *[np.repeat(a[:, None], 1600, axis=1) for a in params], np.random.default_rng(0)), repeats=1)
    out["seconds_per_scenario"] = {"gauss_16": seconds_gauss / n, "monte_carlo_1600": seconds_mc / n}
    plots.quadrature(out, FIGURES / "quadrature_study.png")
    return _write("quadrature", out)


# --- 4. Model selection ------------------------------------------------------

ARCHITECTURES = [(32,), (64, 64), (128, 128, 64), (256, 256, 128)]


def _inference_seconds(predictor, df, repeats=3):
    cat, num = predictor.preprocessor.transform(df)
    with torch.no_grad():
        seconds, _ = _timed(lambda: [m(cat, num) for m in predictor.models], repeats)
    return seconds


def selection():
    """Choose the network size (single networks, raw inputs), then check how much data it needs.

    Selection uses the validation set only. Both validation log loss (which
    would be available with real data) and error against the true
    probability (simulation only) are recorded, to check they agree.

    Rule: the one-standard-error rule (Hastie, Tibshirani and Friedman, *The
    Elements of Statistical Learning*, Section 7.10) - the smallest network
    whose validation log loss is within one standard error of the best. Two
    runs with different data put 64-64 and 128-128-64 in different orders by
    less than 0.001, so a rule that only takes the minimum would pick on noise.
    """
    train_df, val_df, _ = split(dataset(), SEED)
    y, p_true = val_df[LABEL].to_numpy(), val_df["p_true"].to_numpy()
    default_truth = float(simulator.probability(pd.DataFrame([DEFAULTS]))[0])
    architectures, trained = [], {}
    for hidden in ARCHITECTURES:
        start = time.perf_counter()
        predictor, history = train_network(train_df, val_df, hidden=hidden, seed=SEED, log=lambda _: None)
        fit_seconds = time.perf_counter() - start
        trained[hidden] = predictor
        p = predictor.predict(val_df)
        m = metrics(y, p, p_true)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        architectures.append({
            "label": "-".join(map(str, hidden)),
            "hidden": list(hidden),
            "parameters": sum(p.numel() for p in predictor.model.parameters()),
            "flops_per_prediction": nn_flops(hidden),
            "epochs": len(history),
            "fit_seconds": fit_seconds,
            "inference_us_per_prediction": 1e6 * _inference_seconds(predictor, val_df) / len(val_df),
            "val_log_loss": m["log_loss"],
            "val_log_loss_se": float(losses.std(ddof=1) / np.sqrt(len(losses))),
            "val_mae_vs_true": m["mae_vs_true"],
            "default_scenario_error": predictor.predict(DEFAULTS) - default_truth,
        })
        print(architectures[-1], flush=True)

    lowest = min(architectures, key=lambda a: a["val_log_loss"])
    threshold = lowest["val_log_loss"] + lowest["val_log_loss_se"]
    best = min((a for a in architectures if a["val_log_loss"] <= threshold), key=lambda a: a["parameters"])
    chosen = tuple(best["hidden"])

    data_sizes = []
    for size in (10_000, 30_000, 100_000, len(train_df)):
        if size == len(train_df):
            predictor = trained[chosen]
        else:
            predictor, _ = train_network(train_df.iloc[:size], val_df, hidden=chosen, seed=SEED, log=lambda _: None)
        m = metrics(y, predictor.predict(val_df), p_true)
        data_sizes.append({"train_size": size, "val_log_loss": m["log_loss"], "val_mae_vs_true": m["mae_vs_true"]})
        print(data_sizes[-1], flush=True)

    plots.model_selection(architectures, data_sizes, FIGURES / "model_selection.png")
    return _write("selection", {"architectures": architectures, "lowest_log_loss": lowest["label"],
                                "threshold": threshold, "selected": best["label"],
                                "selection_rule": "smallest network within one standard error of the "
                                                  "lowest validation log loss",
                                "default_scenario_truth": default_truth, "data_sizes": data_sizes})


def ablation(seeds=(0, 1, 2, 3, 4)):
    """Physics-informed inputs and ensembling, over five training seeds.

    The selected network is trained five times (seeds 0-4) with and without
    the `DERIVED` energy-cover inputs. For each single run, and for the
    ensemble that averages the five runs' probabilities, this records the
    validation log loss and error against truth, the error at the default
    scenario, and the worst error along the attacker-battery axis through the
    default scenario (30-100 %).

    A first version with three seeds showed that the seed alone moves the
    default-scenario prediction by about 3 points, which is what motivated
    testing an ensemble.
    """
    from .train import HIDDEN
    train_df, val_df, _ = split(dataset(), SEED)
    y, p_true = val_df[LABEL].to_numpy(), val_df["p_true"].to_numpy()
    axis = pd.DataFrame([{**DEFAULTS, "attacker_battery": b} for b in np.arange(30.0, 101.0, 2.5)])
    axis_truth = simulator.probability(axis)
    default_truth = float(simulator.probability(pd.DataFrame([DEFAULTS]))[0])

    def evaluate(p_val, p_default, p_axis):
        m = metrics(y, p_val, p_true)
        return {"val_log_loss": m["log_loss"], "val_mae_vs_true": m["mae_vs_true"],
                "default_scenario_error": float(p_default - default_truth),
                "battery_axis_max_error": float(np.abs(p_axis - axis_truth).max())}

    out = {"hidden": list(HIDDEN), "seeds": list(seeds), "default_scenario_truth": default_truth}
    for name, physics in (("raw inputs only", False), ("with energy-cover inputs", True)):
        runs, preds = [], []
        for seed in seeds:
            predictor, history = train_network(train_df, val_df, hidden=HIDDEN, seed=seed, physics_features=physics,
                                               log=lambda _: None)
            p = (predictor.predict(val_df), predictor.predict(DEFAULTS), predictor.predict(axis))
            preds.append(p)
            runs.append({"seed": seed, "epochs": len(history), **evaluate(*p)})
            print(name, runs[-1], flush=True)
        summary = {key: {"mean": float(np.mean([r[key] for r in runs])), "sd": float(np.std([r[key] for r in runs], ddof=1))}
                   for key in ("val_log_loss", "val_mae_vs_true", "default_scenario_error", "battery_axis_max_error")}
        ensemble = evaluate(*(np.mean([p[i] for p in preds], axis=0) for i in range(3)))
        out[name] = {"runs": runs, "single_model": summary, "ensemble_of_5": ensemble}
        print(name, "ensemble", ensemble, flush=True)
    return _write("ablation", out)


# --- 5. Comparison with alternatives ----------------------------------------

def tyre_mismatch(df):
    """True where either car's tyres do not suit the surface (slicks when damp/wet, treaded tyres when dry)."""
    slicks = ["soft", "medium", "hard"]
    dry = df["track_condition"] == "dry"
    wrong = [(dry & ~df[c].isin(slicks)) | (~dry & df[c].isin(slicks)) for c in ("attacker_tyre", "defender_tyre")]
    return (wrong[0] | wrong[1]).to_numpy()


def _region_errors(test_df, p):
    err = np.abs(p - test_df["p_true"].to_numpy())
    regions = {}
    for column in ("track_condition", "segment"):
        for value in VOCABS[CATEGORICAL[column]]:
            mask = (test_df[column] == value).to_numpy()
            regions[value] = float(err[mask].mean())
    mismatch = tyre_mismatch(test_df)
    regions["tyres_suit_surface"] = float(err[~mismatch].mean())
    regions["tyre_mismatch"] = float(err[mismatch].mean())
    return regions


def _error_quantiles(test_df, p):
    err = np.abs(p - test_df["p_true"].to_numpy())
    return {f"q{int(q * 1000) / 10:g}": float(np.quantile(err, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}


def _model_bytes(obj):
    if isinstance(obj, OvertakePredictor):
        return sum(p.numel() * p.element_size() for m in obj.models for p in m.parameters())
    return len(pickle.dumps(obj.pipeline))


def comparison():
    """Compare the network with the alternatives on the held-out test set.

    Also tests extrapolation: straights of 1300-2000 m, longer than any in
    training, where a data-driven model has no evidence.
    """
    train_df, _, test_df = split(dataset(), SEED)
    y, p_true = test_df[LABEL].to_numpy(), test_df["p_true"].to_numpy()
    network = OvertakePredictor.load()
    models = {"Neural network": network,
              "Logistic regression": baselines.logistic_regression().fit(train_df),
              "Gradient boosting": baselines.gradient_boosting(SEED).fit(train_df)}
    fit_seconds = {"Neural network": torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=True)["fit_seconds"]}
    fit_seconds.update({k: m.fit_seconds for k, m in models.items() if k != "Neural network"})

    sample = test_df.iloc[:10_000]
    out = {"test_attempts": len(test_df), "tyre_mismatch_share": float(tyre_mismatch(test_df).mean()),
           "models": {}, "regions": {}, "error_quantiles": {}}
    for name, model in models.items():
        p = model.predict(test_df)
        seconds, _ = _timed(lambda: model.predict(sample))
        out["models"][name] = {**metrics(y, p, p_true), "fit_seconds": fit_seconds[name],
                               "predict_us_per_scenario": 1e6 * seconds / len(sample),
                               "model_bytes": _model_bytes(model)}
        out["regions"][name] = _region_errors(test_df, p)
        out["error_quantiles"][name] = _error_quantiles(test_df, p)
        err, mismatch = np.abs(p - p_true), tyre_mismatch(test_df)
        out["error_quantiles"][name]["tyre_mismatch_share_of_worst_1pct"] = float(
            mismatch[err >= np.quantile(err, 0.99)].mean())
        out["error_quantiles"][name]["tyre_mismatch_share_of_worst_0.1pct"] = float(
            mismatch[err >= np.quantile(err, 0.999)].mean())

    seconds, p_exact = _timed(lambda: simulator.probability(sample), repeats=1)
    out["models"]["Direct simulation"] = {"mae_vs_true": float(np.abs(p_exact - sample["p_true"]).mean()),
                                          "predict_us_per_scenario": 1e6 * seconds / len(sample),
                                          "note": "the ground truth itself; needs the physical model, "
                                                  "so cannot learn from observed race outcomes"}

    rng = np.random.default_rng(21)
    far = simulator.sample_scenarios(20_000, rng)
    far = far[far["segment"] == "straight"].copy()
    far["run_length_m"] = rng.uniform(1300, 2000, len(far))
    truth = simulator.probability(far)
    out["extrapolation_1300_2000m"] = {name: float(np.abs(m.predict(far) - truth).mean())
                                       for name, m in models.items()}

    plots.method_comparison(out["regions"], FIGURES / "method_comparison.png")
    plots.calibration(y, {k: m.predict(test_df) for k, m in models.items()}, FIGURES / "calibration.png")
    return _write("comparison", out)


# --- 6. Performance ----------------------------------------------------------

def performance():
    """Where the time goes, and what vectorisation and chunking buy.

    * cProfile of simulating 20,000 attempts.
    * Vectorised integration of 200 scenarios together against one at a time.
    * Chunk size for dataset generation (array size vs cache and overhead).
    * FLOPs per prediction and achieved FLOP rate for the network and the
      direct simulation; arithmetic intensity of network inference.
    """
    out = {}
    profiler = cProfile.Profile()
    profiler.enable()
    simulator.generate_dataset(20_000, seed=1)
    profiler.disable()
    stats = pstats.Stats(profiler, stream=io.StringIO()).sort_stats("tottime")
    top = sorted(stats.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:8]
    total = sum(v[2] for v in stats.stats.values())
    out["profile_top_functions"] = [
        {"function": f"{fn[0].split('/')[-1]}:{fn[2]}", "calls": v[1], "own_seconds": v[2],
         "share": v[2] / total} for fn, v in top
    ]

    df = simulator.sample_scenarios(200, np.random.default_rng(4))
    t_vec, _ = _timed(lambda: simulator.closing_run(df))
    t_loop, _ = _timed(lambda: [simulator.closing_run(df.iloc[i:i + 1]) for i in range(len(df))], repeats=1)
    out["vectorisation_200_scenarios"] = {"loop_seconds": t_loop, "vectorised_seconds": t_vec,
                                          "speedup": t_loop / t_vec}

    out["chunk_size_40k_scenarios"] = []
    for chunk in (1_000, 5_000, 20_000, 40_000):
        seconds, _ = _timed(lambda: simulator.generate_dataset(40_000, seed=2, chunk=chunk), repeats=1)
        out["chunk_size_40k_scenarios"].append({"chunk": chunk, "seconds": seconds})

    sample = simulator.sample_scenarios(10_000, np.random.default_rng(5))
    t_sim, _ = _timed(lambda: simulator.probability(sample), repeats=1)
    run = simulator.closing_run(sample)[0]
    steps = run.time_s / simulator.TIME_STEP_S
    sim_flops = float(np.mean(steps) * FLOPS_PER_RK4_STEP + FLOPS_QUADRATURE)
    network = OvertakePredictor.load()
    hidden = network.model.hidden
    t_nn = _inference_seconds(network, sample)
    nn = nn_flops(hidden) * len(network.models)
    params = network.n_parameters
    width_in = network.model.mlp[0].in_features
    out["flops"] = {
        "network_hidden": list(hidden),
        "network_members": len(network.models),
        "network_flops_per_prediction": nn,
        "network_gflops_achieved": nn * len(sample) / t_nn / 1e9,
        "network_us_per_prediction_batch_10k": 1e6 * t_nn / len(sample),
        "simulation_mean_rk4_steps": float(np.mean(steps)),
        "simulation_flops_per_scenario_estimate": sim_flops,
        "simulation_gflops_achieved": sim_flops * len(sample) / t_sim / 1e9,
        "simulation_us_per_scenario": 1e6 * t_sim / len(sample),
        # Weights (4 bytes each) are read once per batch; each prediction reads
        # its inputs and writes one output.
        "network_arithmetic_intensity_flop_per_byte": {
            str(b): nn * b / (4 * params + 4 * b * (width_in + 1)) for b in (1, 64, 1024, 10_000)},
    }
    out["memory"] = {
        "network_parameter_bytes": 4 * params,
        "dataset_300k_bytes": int(dataset().memory_usage(deep=True).sum()),
        "ode_state_bytes_per_20k_chunk": 6 * 8 * 20_000,
    }
    return _write("performance", out)


# --- 7. Sensitivity ----------------------------------------------------------

SENSITIVITY_RANGES = {
    "gap_s": (0.3, 1.5, "0.3 s", "1.5 s"),
    "run_length_m": (400, 1200, "400 m", "1200 m"),
    "attacker_battery": (20, 100, "20%", "100%"),
    "defender_battery": (100, 20, "100%", "20%"),
    "attacker_tyre_age": (30, 0, "30 laps", "new"),
    "defender_tyre_age": (0, 30, "new", "30 laps"),
    "attacker_skill": (65, 95, "65", "95"),
    "defender_skill": (95, 65, "95", "65"),
    "top_speed_delta_kph": (-10, 10, "-10 km/h", "+10 km/h"),
    "pace_delta_s": (-1.0, 1.0, "-1 s", "+1 s"),
    "track_temp_c": (20, 50, "20 C", "50 C"),
    "attacker_boost": (False, True, "off", "on"),
    "defender_boost": (True, False, "on", "off"),
    "overtake_mode": (False, True, "off", "on"),
    "circuit": ("monaco", "bahrain", "Monaco", "Bahrain"),
}


def sensitivity():
    """Vary one input at a time between plausible extremes around the default scenario.

    Compares the network's response with the simulator's, which checks that
    the network learned the right sensitivities and not just the right
    average. One-at-a-time analysis misses interactions; the what-if figure
    shows two of the important ones.
    """
    network = OvertakePredictor.load()
    rows = []
    for column, (low, high, low_label, high_label) in SENSITIVITY_RANGES.items():
        scenarios = [{**DEFAULTS, column: low}, {**DEFAULTS, column: high}]
        if column == "gap_s":
            for s in scenarios:
                s["overtake_mode"] = s["gap_s"] <= 1.0
        exact = simulator.probability(pd.DataFrame(scenarios))
        rows.append({"input": column, "low": low_label, "high": high_label,
                     "network": [float(v) for v in network.predict(scenarios)],
                     "exact": [float(v) for v in exact]})
    rows.sort(key=lambda r: abs(r["exact"][1] - r["exact"][0]), reverse=True)
    plots.sensitivity(rows, FIGURES / "sensitivity.png")
    plots.what_if(network.predict, FIGURES / "what_if.png",
                  exact=lambda s: simulator.probability(pd.DataFrame(s)))
    base = {"network": network.predict(DEFAULTS), "exact": float(simulator.probability(pd.DataFrame([DEFAULTS]))[0])}

    # The bias near a full attacker battery, repeated on a wet track with intermediates.
    wet = {**DEFAULTS, "track_condition": "wet", "attacker_tyre": "intermediate",
           "defender_tyre": "intermediate", "track_temp_c": 20.0}
    axis = pd.DataFrame([{**wet, "attacker_battery": b} for b in (40.0, 60.0, 80.0, 100.0)])
    wet_axis = [{"attacker_battery": b, "network": float(n), "exact": float(e)}
                for b, n, e in zip(axis["attacker_battery"], network.predict(axis), simulator.probability(axis))]
    _, _, test_df = split(dataset(), SEED)
    mask = ((test_df["track_condition"] == "wet") & (test_df["segment"] == "straight")
            & test_df["attacker_tyre"].isin(["intermediate", "wet"]) & test_df["defender_tyre"].isin(["intermediate", "wet"]))
    err = network.predict(test_df[mask]) - test_df.loc[mask, "p_true"].to_numpy()
    wet_test = {"scenarios": int(mask.sum()), "mean_signed_error": float(err.mean()),
                "mean_abs_error": float(np.abs(err).mean())}
    return _write("sensitivity", {"default_scenario": base, "rows": rows,
                                  "wet_intermediates_battery_axis": wet_axis,
                                  "test_wet_straights_treaded_tyres": wet_test})


EXPERIMENTS = {
    "solver": solver,
    "timestep": timestep,
    "quadrature": quadrature,
    "selection": selection,
    "ablation": ablation,
    "comparison": comparison,
    "performance": performance,
    "sensitivity": sensitivity,
}


def main(argv=None):
    names = (argv if argv is not None else sys.argv[1:]) or ["all"]
    if names == ["all"]:
        names = list(EXPERIMENTS)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names:
        start = time.perf_counter()
        print(f"== {name}", flush=True)
        result = EXPERIMENTS[name]()
        print(json.dumps(result, indent=1, default=float)[:3000], flush=True)
        print(f"== {name} done in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
