"""Experiments behind the project documentation.

    python -m overtake.experiments all
    python -m overtake.experiments solver timestep      # a subset

``all``, also the default when no name is given, runs every experiment in
the table below except ``figures``, in table order. Every experiment except
``diagram`` and ``figures`` saves its numbers as
``reports/results/<name>.json`` (``real_data.json`` for ``real``). All except
``adaptive``, ``ablation``, ``baselines`` and ``performance`` also draw
figures in ``reports/figures/``.

| Name | What it does |
|---|---|
| `solver` | Euler and RK4 against analytical solutions; observed order |
| `timestep` | Error and cost of the full two-car model against time step; linear vs Hermite crossing |
| `stability` | Explicit-solver stability on the stiffest part of the model (battery running out) |
| `adaptive` | Fixed-step vectorised RK4 against SciPy's adaptive RK45 with event location |
| `quadrature` | Quadrature rules and Monte Carlo against a reference |
| `speed_trace` | Plausibility of the car model: speed along a straight for each deployment |
| `constants` | Sensitivity of the exact probability to the estimated physical and outcome constants |
| `rules` | Sensitivity of the exact probability to the Overtake Mode rules (energy and taper) |
| `selection` | Network size and training-set size, on the validation set |
| `ablation` | Physics-informed inputs and five-network ensembles, five seeds |
| ``baselines`` (`tune_baselines`) | Tuning of the two baseline models on the validation set |
| `comparison` | Network vs logistic regression vs gradient boosting vs direct simulation, on the test set |
| `performance` | Profiling, vectorisation, chunk size, FLOPs and memory |
| `sensitivity` | One-at-a-time sensitivity, network vs simulator truth |
| `real` | Comparison with real 2026 races from OpenF1 (speeds, overtakes, gaps) |
| `diagram` | The workflow overview figure |
| `figures` | Redraw the figures whose data are saved in ``reports/results`` (no recomputation) |

``baselines``, ``comparison``, ``performance`` and ``sensitivity`` load the
trained network from ``models/``, so run ``python -m overtake.train`` first.
The experiments that train or score models share one simulated dataset per
process (`dataset`): with the defaults, the same 300,000 attempts and the
same split that ``python -m overtake.train`` uses, so the test set is unseen
by the saved network.
"""
import contextlib
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

from . import baselines, physics, plots, simulator
from .features import CATEGORICAL, DEFAULTS, LABEL, VOCABS
from .model import OvertakeNet
from .physics import CarParameters, CarState, acceleration, euler_step, integrate, rk4_step, two_car_rhs
from .predict import DEFAULT_MODEL, OvertakePredictor
from .train import REPORTS, metrics, split, train_network

FIGURES, RESULTS = REPORTS / "figures", REPORTS / "results"
# Seed of the shared dataset, its split and the models trained here; 0 matches python -m overtake.train.
SEED = 0


def _write(name, result):
    """Save ``result`` as ``reports/results/<name>.json`` and return it unchanged."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    # default=float converts the NumPy scalars json cannot serialise (int64, float32, bool_).
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def _timed(func, repeats=3):
    """Best wall-clock time (s) of ``repeats`` calls of ``func``, and the last call's result."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = func()
        # The fastest run is the one least disturbed by other work on the machine.
        best = min(best, time.perf_counter() - start)
    return best, out


@functools.lru_cache(maxsize=None)
def dataset(n=300_000, seed=SEED):
    """Simulated attempts, generated once per process and shared by the experiments.

    The cache returns the same DataFrame object to every caller, so callers
    must not modify it in place.

    Parameters
    ----------
    n : int, optional
        Number of attempts. The default matches ``python -m overtake.train``.
    seed : int, optional
        Seed of `overtake.simulator.generate_dataset`.

    Returns
    -------
    pandas.DataFrame
        Scenarios in the `overtake.features` schema plus the 0/1 ``overtake``
        label and the exact probability ``p_true``.
    """
    return simulator.generate_dataset(n, seed=seed)


@contextlib.contextmanager
def _patched(module, **values):
    """Temporarily replace module-level constants, restoring them afterwards.

    Used only by the sensitivity studies of the model's constants and rules
    (`constants`, `rules`); the rest of the code never changes them.

    Parameters
    ----------
    module : module
        For example `overtake.simulator`.
    **values
        Attribute name -> temporary value.

    Yields
    ------
    None
    """
    old = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(module, name, value)


def nn_flops(hidden, n_inputs=None):
    """Floating-point operations for one prediction by one network.

    Counts a multiply-add as 2 FLOPs, a bias add as 1 and a SiLU activation as
    about 5 (exponential, add, divide, multiply). Embedding look-ups are memory
    reads, not arithmetic, and dropout is inactive at inference, so neither is
    counted.

    Parameters
    ----------
    hidden : tuple of int
        Width of each hidden layer, e.g. ``(64, 64)``.
    n_inputs : int, optional
        Width of the first layer's input (embeddings + numeric inputs). By
        default that of an `overtake.model.OvertakeNet` on the raw inputs
        without the derived energy-cover inputs: 33 (19 embedding values,
        11 numbers and 3 flags).

    Returns
    -------
    int
        FLOPs per prediction for a single network; multiply by the number of
        ensemble members for the whole predictor.

    Examples
    --------
    >>> nn_flops((64, 64))
    13313
    """
    if n_inputs is None:
        n_inputs = OvertakeNet(hidden).mlp[0].in_features
    # Layer widths from input to the single output logit; consecutive pairs are the linear layers.
    widths = [n_inputs, *hidden, 1]
    linear = sum(2 * a * b + b for a, b in zip(widths[:-1], widths[1:]))
    return linear + 5 * sum(hidden)


# Hand count (estimates) of the arithmetic per scenario, for the simulator's
# FLOP rate in `performance`. physics.two_car_rhs: about 35 FLOPs per car in
# `acceleration` plus about 20 for the slipstream (an exponential counted as
# ~10). An RK4 step makes 4 right-hand-side evaluations plus ~14 element-wise
# operations (stage sums and the weighted update) on each of the 6 state rows.
FLOPS_PER_RHS = 2 * 35 + 20
FLOPS_PER_RK4_STEP = 4 * FLOPS_PER_RHS + 6 * 14
# 16 Gauss nodes at ~30 FLOPs each (normal density + logistic) and ~40 for the closed-form tail (estimates).
FLOPS_QUADRATURE = 16 * 30 + 40


# --- 1. Solver verification -------------------------------------------------

def _single_car(car):
    """Right-hand side ``f([x, v, e])`` of one car on its own, with grip and electrical power switched off."""
    # A huge friction coefficient removes the traction limit, and no deployment or energy removes the
    # MGU-K, so only the engine power and the resistances in ``car`` act.
    state = CarState(mu=np.array([1e9]), cda_m2=np.array([car.cda_m2]),
                     deploy_share=np.array([0.0]), energy_j=np.array([0.0]))

    def f(y):
        """Time derivative of the single-car state [position m, speed m/s, energy used J]."""
        accel, ers = acceleration(y[1:2], y[2:3], state, state.cda_m2, car)
        return np.array([y[1], accel[0], ers[0]])
    return f


def solver():
    """Verify Euler and RK4 on two special cases of the car model with exact solutions.

    * Constant power ``P``, no resistance (``m dv/dt = P / v``):
      ``v^2 = v0^2 + 2 P t / m`` and ``x = m / (3P) (v^3 - v0^3)``.
    * Drag only (``m dv/dt = -k v^2`` with ``k = rho CdA / 2``):
      ``v = v0 / (1 + k v0 t / m)`` and ``x = (m / k) ln(1 + k v0 t / m)``.

    Both start at 30 m/s and run for 12.8 s, a whole number of every step
    tested (0.8 s halving down to 0.0125 s). The observed order of
    convergence is the slope of log(error) against log(dt) where the error
    is in its asymptotic regime and above round-off: the four smallest steps
    for Euler, and 0.4-0.05 s for RK4, whose drag-only errors at the two
    smallest steps are near double-precision round-off.

    Returns
    -------
    dict
        Per problem: the steps, the exact final position (m), the absolute
        final-position error (m) of each method at each step, and each
        method's observed order.
    """
    m, p, k, v0, t_end = 800.0, 500e3, 0.6, 30.0, 12.8
    power_only = replace(CarParameters(), mass_kg=m, ice_power_w=p, cda_m2=0.0, cla_m2=0.0,
                         rolling_coeff=0.0, drivetrain_efficiency=1.0)
    # CdA chosen so that 1/2 rho CdA equals k with rho = 1.2 kg/m^3.
    drag_only = replace(power_only, ice_power_w=0.0, cda_m2=2 * k / 1.2, air_density=1.2)
    problems = {
        # (v0^2 + 2 P t / m)^1.5 is v^3 at t_end.
        "Constant power, no drag": (power_only, m / (3 * p) * ((v0**2 + 2 * p * t_end / m) ** 1.5 - v0**3)),
        "Drag only, no power": (drag_only, m / k * np.log(1 + k * v0 * t_end / m)),
    }
    dts = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    out = {}
    for title, (car, x_exact) in problems.items():
        r = {"dt": dts, "exact_position_m": x_exact}
        for method in ("euler", "rk4"):
            # Index 0 of the final state is the position; the state starts at [x = 0, v = v0, e = 0].
            r[method] = [abs(integrate(_single_car(car), [0.0, v0, 0.0], dt, int(round(t_end / dt)), method)[0]
                             - x_exact) for dt in dts]
            # Euler: the four smallest steps (at larger ones its constant-power error is not yet
            # asymptotic). RK4: 0.4-0.05 s,
            # leaving out the two smallest steps, whose drag-only errors are near round-off.
            tail = slice(-4, None) if method == "euler" else slice(1, 5)
            r[method + "_order"] = float(np.polyfit(np.log(dts[tail]), np.log(r[method][tail]), 1)[0])
        out[title] = r
    plots.convergence(out, FIGURES / "solver_convergence.png")
    return _write("solver", out)


# --- 2. Time-step study ------------------------------------------------------

def timestep(n=2000, dts=(0.4, 0.2, 0.1, 0.05, 0.025, 0.0125), reference_dt=0.001, fit_max_dt=0.1):
    """Deficit error of the full model against a fine-step RK4 reference.

    The full model has kinks (traction limit, MGU-K taper, energy running
    out), so the observed order is lower than the smooth-case order; this
    study chooses the step that keeps the error well below the 1 ms
    acceptance limit (0.5 % of the 0.2 s execution noise).

    Also measured:

    * the observed order, fitted only over steps up to ``fit_max_dt``: larger
      steps (0.2 and 0.4 s) are beyond the stability limits on the battery
      ramp (0.114 s Euler, 0.159 s RK4; see `stability`), where the error is
      not in its asymptotic regime;
    * RK4 with straight-line instead of Hermite interpolation at the braking
      point, to measure what the fourth-order crossing buys;
    * the step and run time Euler would need, from a power-law fit of its
      worst error over the same steps, to meet the 1 ms limit and to match
      RK4's worst error at the chosen step.

    Parameters
    ----------
    n : int
        Random scenarios (`overtake.simulator.sample_scenarios`, seed 7).
    dts : sequence of float
        Steps tested, s.
    reference_dt : float
        Step of the RK4 reference solution, s.
    fit_max_dt : float
        Largest step included in the order fits, s.

    Returns
    -------
    dict
        Per method: worst and mean deficit error (s) and run time (s) per
        step, and the observed order of the mean and of the worst error; the
        same for RK4 with linear crossing; and, for each Euler target, the
        target error (s), the step needed (s) and its run time relative to
        RK4 at the chosen step.

    Raises
    ------
    ValueError
        If ``dts`` does not contain the simulator's step,
        `overtake.simulator.TIME_STEP_S`.
    """
    df = simulator.sample_scenarios(n, np.random.default_rng(7))
    reference = simulator.closing_run(df, dt=reference_dt)[0].deficit_s
    # The small tolerance keeps a step equal to fit_max_dt despite floating-point representation.
    fit = [i for i, dt in enumerate(dts) if dt <= fit_max_dt + 1e-12]
    log_dt = np.log(np.asarray(dts)[fit])
    out = {"dt": list(dts), "reference_dt": reference_dt, "scenarios": n, "fit_dt": [dts[i] for i in fit]}
    for name, method, crossing in (("rk4", "rk4", "hermite"), ("euler", "euler", "hermite"),
                                   ("rk4_linear_crossing", "rk4", "linear")):
        rows = {"max_error_s": [], "mean_error_s": [], "seconds": []}
        for dt in dts:
            # closing_run returns (run, grip_a, grip_d); only the run is needed.
            seconds, (run, _, _) = _timed(
                lambda: simulator.closing_run(df, dt=dt, method=method, crossing=crossing), repeats=2)
            err = np.abs(run.deficit_s - reference)
            rows["max_error_s"].append(float(err.max()))
            rows["mean_error_s"].append(float(err.mean()))
            rows["seconds"].append(seconds)
        # Observed order: slope of log(error) against log(dt) over the fitted steps.
        for key, errors in (("observed_order", "mean_error_s"), ("observed_order_max_error", "max_error_s")):
            rows[key] = float(np.polyfit(log_dt, np.log(np.asarray(rows[errors])[fit]), 1)[0])
        out[name] = rows

    # Euler's worst error as a power law of the step, err = exp(c) dt^p, fitted over the
    # same steps; its run time scales as 1/dt, so seconds x step is roughly constant.
    chosen = list(dts).index(simulator.TIME_STEP_S)
    p, c = np.polyfit(log_dt, np.log(np.asarray(out["euler"]["max_error_s"])[fit]), 1)
    finest = int(np.argmin(dts))
    # Seconds x step at the finest step, where the fixed per-run overheads matter least.
    euler_seconds_times_dt = out["euler"]["seconds"][finest] * dts[finest]
    rk4_seconds = out["rk4"]["seconds"][chosen]
    out["euler_equivalent"] = {}
    for key, target in (("for_1ms", 1e-3), ("to_match_rk4", out["rk4"]["max_error_s"][chosen])):
        # Invert the power law: dt = exp((log(err) - c) / p).
        dt_needed = float(np.exp((np.log(target) - c) / p))
        out["euler_equivalent"][key] = {"target_error_s": float(target), "dt_s": dt_needed,
                                        "cost_vs_rk4": float(euler_seconds_times_dt / dt_needed / rk4_seconds)}
    plots.timestep(out, FIGURES / "timestep_study.png", simulator.TIME_STEP_S)
    return _write("timestep", out)


def rk4_amplification(z):
    """RK4 amplification factor for the test equation ``y' = lambda y``.

    One RK4 step multiplies the solution by ``R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24``
    with ``z = lambda dt``; the step is stable while ``|R(z)| <= 1``.

    Parameters
    ----------
    z : float or numpy.ndarray

    Returns
    -------
    float or numpy.ndarray
    """
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def stability(dts=(0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3), battery_j=0.2e6, t_end=3.0,
              reference_dt=0.001):
    """Stability of Euler and RK4 on the stiffest part of the model: the battery running out.

    In the last ``energy_smoothing_j`` (20 kJ) of a battery the MGU-K power is
    ramped down in proportion to the energy left, so the energy used ``e``
    obeys ``de/dt = s P (E - e) / S`` (deployment share ``s``, MGU-K limit
    ``P``, energy available ``E``, ramp width ``S``). That is exponential
    decay with rate ``lambda = s P / S``: 17.5 1/s with Boost, against about
    0.25 1/s for the speed (drag and power). An explicit method is stable only
    while ``lambda dt`` lies inside its stability region, which on the real
    axis is 2 for Euler and about 2.785 for RK4, so this ramp, not the vehicle
    dynamics, sets the largest usable time step.

    The check: one car starts at 200 km/h with only ``battery_j`` of energy
    (so the ramp is reached within a second) and is integrated for ``t_end``
    seconds at each step. An unstable step shows as the energy used
    overshooting the energy available, a physical constraint the exact
    solution never breaks. Positions are compared with a ``reference_dt``
    solution.

    Parameters
    ----------
    dts : sequence of float
        Time steps to test, s.
    battery_j : float
        Energy available, J.
    t_end : float
        Integration time, s; must be a whole number of every step in ``dts``
        so that positions are compared at the same instant.
    reference_dt : float
        Step of the reference solution, s.

    Returns
    -------
    dict
        Predicted step limits per deployment and method, and per step the
        largest energy overshoot (J) and final position error (m).
    """
    from scipy.optimize import brentq

    steps = {dt: int(round(t_end / dt)) for dt in dts}
    if any(abs(n * dt - t_end) > 1e-9 for dt, n in steps.items()):
        raise ValueError("t_end must be a whole number of every time step")
    car = CarParameters()
    # Where |R(z)| = 1 on the negative real axis.
    rk4_limit = -brentq(lambda z: rk4_amplification(z) - 1, -3.0, -2.0)
    v0 = 200 / 3.6
    out = {"dt": list(dts), "battery_j": battery_j, "t_end_s": t_end, "chosen_dt": simulator.TIME_STEP_S,
           "real_axis_limit": {"euler": 2.0, "rk4": rk4_limit}, "deployments": {}}
    for title, share in (("Boost", 1.0), ("Standard map", simulator.STANDARD_MAP_SHARE)):
        state = CarState(mu=np.array([car.mu_reference]), cda_m2=np.array([car.cda_m2]),
                         deploy_share=np.array([share]), energy_j=np.array([battery_j]))

        def f(y, state=state):
            """One car alone, state [x, v, e]; ``state`` is bound here so each deployment keeps its own."""
            accel, ers = acceleration(y[1:2], y[2:3], state, state.cda_m2, car)
            return np.array([y[1], accel[0], ers[0]])

        rate = share * car.mguk_power_w / car.energy_smoothing_j
        reference = integrate(f, [0.0, v0, 0.0], reference_dt, int(round(t_end / reference_dt)))
        row = {"share": share, "decay_rate_per_s": rate,
               "predicted_dt_limit_s": {"euler": 2.0 / rate, "rk4": rk4_limit / rate}}
        for method, step in (("euler", euler_step), ("rk4", rk4_step)):
            overshoot, position_error = [], []
            for dt in dts:
                y, worst = np.array([0.0, v0, 0.0]), 0.0
                for _ in range(steps[dt]):
                    y = step(f, y, dt)
                    worst = max(worst, y[2] - battery_j)  # energy used beyond what the battery holds
                overshoot.append(float(worst))
                position_error.append(float(abs(y[0] - reference[0])))
            row[method] = {"max_energy_overshoot_j": overshoot, "final_position_error_m": position_error}
        out["deployments"][title] = row
    plots.stability(out, FIGURES / "stability.png")
    return _write("stability", out)


def adaptive(n=100, rtols=(1e-3, 1e-6, 1e-9), reference_dt=0.001, timing_n=20_000):
    """Fixed-step vectorised RK4 against SciPy's adaptive RK45 with event location.

    An adaptive solver chooses its own steps to meet a tolerance and stops
    exactly at the braking point (an event), so it needs neither a time-step
    study nor the Hermite crossing. SciPy's ``solve_ivp`` integrates one
    scenario at a time, however, while the fixed-step RK4 integrates all of
    them together as arrays. Both are scored on the same scenarios against
    the 0.001 s RK4 reference used by `timestep`.

    Parameters
    ----------
    n : int
        Scenarios for the accuracy comparison: the first ``n`` of the
        `timestep` sample (seed 7).
    rtols : sequence of float
        Relative tolerances given to ``solve_ivp``.
    reference_dt : float
        Step of the RK4 reference, s.
    timing_n : int
        Scenarios integrated together to time the fixed-step RK4 per scenario,
        as in dataset generation (with only ``n`` the fixed cost of each
        vectorised step would dominate).

    Returns
    -------
    dict
        For the fixed-step RK4 and for each tolerance: worst deficit error
        (s), run time per scenario (microseconds) and right-hand-side
        evaluations per scenario.
    """
    from scipy.integrate import solve_ivp

    df = simulator.sample_scenarios(2000, np.random.default_rng(7)).iloc[:n].reset_index(drop=True)
    reference = simulator.closing_run(df, dt=reference_dt)[0].deficit_s
    run = simulator.closing_run(df)[0]
    many = simulator.sample_scenarios(timing_n, np.random.default_rng(8))
    seconds, _ = _timed(lambda: simulator.closing_run(many))
    out = {"scenarios": n, "reference_dt": reference_dt, "timing_scenarios": timing_n,
           "rk4_fixed_step": {"dt": simulator.TIME_STEP_S,
                              "max_error_s": float(np.abs(run.deficit_s - reference).max()),
                              "us_per_scenario": 1e6 * seconds / timing_n,
                              # Four right-hand-side evaluations per step, plus two at the crossing.
                              "rhs_evaluations_per_scenario": float(4 * np.mean(run.time_s) / simulator.TIME_STEP_S + 2)},
           "solve_ivp_rk45": []}

    ic = simulator.initial_conditions(df)
    # The states have different units and sizes (m, m/s, J); energy is about a
    # thousand times larger in magnitude, so its absolute tolerance is too.
    scale = np.array([1.0, 1.0, 1e3, 1.0, 1.0, 1e3])

    def deficit(i, rtol):
        """Deficit (s) at the braking point of scenario ``i`` from ``solve_ivp``, and its RHS evaluations."""
        attacker, defender = ic["attacker"].take([i]), ic["defender"].take([i])
        length = ic["length"][i]

        def rhs(t, y):
            """Two-car right-hand side for one scenario in the (t, y) form ``solve_ivp`` expects."""
            return two_car_rhs(y[:, None], attacker, defender, simulator.CAR)[:, 0]

        def braking_point(t, y):
            """Event function: zero when the defender reaches the braking point."""
            return y[3] - length
        braking_point.terminal, braking_point.direction = True, 1

        y0 = [-ic["gap"][i] * ic["v0_d"][i], ic["v0_a"][i], 0.0, 0.0, ic["v0_d"][i], 0.0]
        sol = solve_ivp(rhs, (0.0, 60.0), y0, method="RK45", rtol=rtol, atol=rtol * scale, events=braking_point)
        xa, va, _, xd, _, _ = sol.y_events[0][0]
        return (xd - xa) / max(va, 1.0), sol.nfev

    for rtol in rtols:
        start = time.perf_counter()
        results = [deficit(i, rtol) for i in range(n)]
        elapsed = time.perf_counter() - start
        values = np.array([r[0] for r in results])
        out["solve_ivp_rk45"].append({"rtol": rtol, "max_error_s": float(np.abs(values - reference).max()),
                                      "us_per_scenario": 1e6 * elapsed / n,
                                      "rhs_evaluations_per_scenario": float(np.mean([r[1] for r in results]))})
    return _write("adaptive", out)


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


# --- 4. Plausibility and sensitivity of the physical model ------------------------

SPEED_TRACE_DEPLOYMENTS = {
    "Boost and Overtake Mode": (1.0, True),
    "Boost": (1.0, False),
    "Standard map": (simulator.STANDARD_MAP_SHARE, False),
    "No electrical power": (0.0, False),
}


def speed_trace(distances_m=(400, 800, 1300), duration_s=120.0):
    """Speed reached along a straight for each deployment, and the two-car speed traces.

    One car on new soft tyres in the dry (relative grip 1) leaves the corner
    at the simulator's exit speed with a full battery and accelerates for
    ``duration_s`` (`overtake.realdata.simulate_run`: the same force model
    and RK4). Its speed at fixed distances is what the documentation compares
    with real speed traps. By the end of the run every battery is empty, so
    the final speed is the engine-only top speed; the peak speed is reached
    while electrical power lasts. The figure shows both cars along the
    default scenario's 800 m straight, and with a defender whose 30 % charge
    runs out mid-straight.

    Parameters
    ----------
    distances_m : sequence of float
        Distances from the corner exit at which the speed is reported, m.
    duration_s : float
        Length of the single-car runs, s.

    Returns
    -------
    dict
        ``exit_speed_kph``; per deployment the speed (km/h) at each distance,
        the peak speed, the speed after ``duration_s``, the speed below which
        traction rather than power limits acceleration, and the fastest decay
        rate of the speed equation (-d(acceleration)/dv, 1/s); and both cars'
        speeds at the last step before the braking point in the two figure
        scenarios.
    """
    from .predict import complete
    from .realdata import simulate_run

    exit_kph = 3.6 * simulator.EXIT_SPEED_MPS
    out = {"exit_speed_kph": exit_kph, "distances_m": list(distances_m), "duration_s": duration_s,
           "deployments": {}}
    for name, (share, overtake_mode) in SPEED_TRACE_DEPLOYMENTS.items():
        run = simulate_run(exit_kph, duration_s, share, overtake_mode=overtake_mode, battery=1.0)
        out["deployments"][name] = {
            "deploy_share": share, "overtake_mode": overtake_mode,
            # Position increases monotonically, so speed can be interpolated against it.
            "speed_kph_at_m": {str(d): float(np.interp(d, run["x"], run["kph"])) for d in distances_m},
            "peak_speed_kph": float(run["kph"].max()),
            "speed_kph_after_duration": float(run["kph"].iloc[-1]),
            **_speed_dynamics(share, overtake_mode),
        }
    default = complete({}).iloc[0].to_dict()
    traces = {"Default scenario": simulator.speed_trace(default),
              "Defender with 30 % charge": simulator.speed_trace({**default, "defender_battery": 30.0})}
    out["end_of_run_kph"] = {title: {"attacker": float(tr["attacker_kph"].iloc[-1]),
                                     "defender": float(tr["defender_kph"].iloc[-1])}
                             for title, tr in traces.items()}
    plots.speed_trace(traces, FIGURES / "speed_trace.png")
    return _write("speed_trace", out)


def _speed_dynamics(share, overtake_mode, car=simulator.CAR):
    """Where traction limits one car on new softs in the dry, and how fast its speed equation decays.

    Parameters
    ----------
    share : float
        MGU-K deployment share (1 Boost, 0.55 standard map, 0 none).
    overtake_mode : bool
        Whether the Overtake Mode taper applies.
    car : CarParameters, optional

    Returns
    -------
    dict
        ``traction_limited_below_kph``: speed where the power-limited and
        traction-limited drive forces are equal; ``max_speed_decay_rate_per_s``:
        the largest -d(acceleration)/dv between 150 and 350 km/h with energy
        available, the stiffness of the speed equation that `stability`
        compares with the battery ramp.
    """
    from scipy.optimize import brentq

    state = CarState(mu=np.array([car.mu_reference]), cda_m2=np.array([car.cda_m2]),
                     deploy_share=np.array([share]), energy_j=np.array([1e9]),
                     overtake_mode=np.array([overtake_mode]))

    def power_minus_traction(v):
        """Power-limited minus traction-limited drive force at speed ``v`` (m/s), N."""
        p_k = share * physics.mguk_power_limit(np.array([v]), np.array([overtake_mode]), car)[0]
        f_power = car.drivetrain_efficiency * (car.ice_power_w + p_k) / v
        f_traction = car.mu_reference * car.driven_axle_share * (
            car.mass_kg * physics.GRAVITY + 0.5 * car.air_density * car.cla_m2 * v**2)
        return f_power - f_traction

    def accel(v):
        """Acceleration (m/s^2) at speed ``v`` with no energy used yet."""
        return acceleration(np.array([v]), np.array([0.0]), state, state.cda_m2, car)[0][0]

    speeds = np.linspace(150, 350, 401) / 3.6
    h = 1e-3  # m/s; central difference of the acceleration
    rates = [-(accel(v + h) - accel(v - h)) / (2 * h) for v in speeds]
    return {"traction_limited_below_kph": 3.6 * brentq(power_minus_traction, 10.0, 90.0),
            "max_speed_decay_rate_per_s": float(max(rates))}


# Estimated (not regulated) constants varied by `constants`: label, where it lives, name.
ESTIMATED_CONSTANTS = [
    ("Corner-exit speed", "simulator", "EXIT_SPEED_MPS"),
    ("Execution noise", "simulator", "EXECUTION_SD"),
    ("Lunge range", "simulator", "LUNGE_RANGE_S"),
    ("Margin logit slope", "simulator", "MARGIN_LOGIT_PER_S"),
    ("Usable battery energy", "simulator", "BATTERY_USABLE_J"),
    ("Standard-map share", "simulator", "STANDARD_MAP_SHARE"),
    ("Mass", "car", "mass_kg"),
    ("Drag area", "car", "cda_m2"),
    ("Downforce area", "car", "cla_m2"),
    ("Tyre friction", "car", "mu_reference"),
    ("Engine power", "car", "ice_power_w"),
    ("Slipstream drag reduction", "car", "tow_max"),
    ("Rear-axle load share", "car", "driven_axle_share"),
]


def constants(change=0.1, n=2000):
    """How much the exact probability depends on each estimated constant.

    Each constant that is an estimate rather than a rule (`ESTIMATED_CONSTANTS`)
    is moved down and then up by ``change`` (10 %), the others staying at
    their values, and the exact probability is recomputed for the default
    scenario and for ``n`` random scenarios. This answers which assumptions
    matter most, which the input sensitivity (`sensitivity`) cannot.

    Parameters
    ----------
    change : float
        Relative change applied to each constant.
    n : int
        Random scenarios (`overtake.simulator.sample_scenarios`, seed 31).

    Returns
    -------
    dict
        Per constant: its value, the default-scenario probability at the low
        and high value, and the mean absolute change in probability over the
        random scenarios at each; sorted by the larger mean change.
    """
    car = simulator.CAR
    scenarios = pd.concat([pd.DataFrame([DEFAULTS]), simulator.sample_scenarios(n, np.random.default_rng(31))],
                          ignore_index=True)
    base = simulator.probability(scenarios)

    def scaled(value, factor):
        """``value`` times ``factor``; for a table (dict) every entry is scaled."""
        # EXECUTION_SD is a table by track condition; scale every entry.
        return {k: v * factor for k, v in value.items()} if isinstance(value, dict) else value * factor

    rows = []
    for label, where, name in ESTIMATED_CONSTANTS:
        value = getattr(car, name) if where == "car" else getattr(simulator, name)
        probabilities = []
        for factor in (1 - change, 1 + change):
            if where == "car":
                probabilities.append(simulator.probability(scenarios, car=replace(car, **{name: value * factor})))
            else:
                with _patched(simulator, **{name: scaled(value, factor)}):
                    probabilities.append(simulator.probability(scenarios))
        rows.append({"constant": label, "name": name, "value": value,
                     "default_scenario": [float(q[0]) for q in probabilities],
                     "mean_abs_change": [float(np.abs(q[1:] - base[1:]).mean()) for q in probabilities]})
    rows.sort(key=lambda r: max(r["mean_abs_change"]), reverse=True)
    out = {"change": change, "scenarios": n, "default_scenario": float(base[0]), "rows": rows}
    plots.constants(out, FIGURES / "constants.png")
    return _write("constants", out)


def rules(energies_mj=(0.0, 0.25, 0.5, 0.75, 1.0), taper_starts_kph=(290, 300, 310, 320, 330, 337, 345, 350),
          batteries=(30, 60, 100)):
    """How the exact probability responds to the two Overtake Mode parameters.

    Overtake Mode gives the attacker extra energy (0.5 MJ) and moves the start
    of the MGU-K taper from 290 to 337 km/h. Each is varied with the other at
    its 2026 value, for the default scenario at several attacker charge
    levels: the question a rule maker would ask of the model.

    Parameters
    ----------
    energies_mj : sequence of float
        Overtake Mode extra energy, MJ.
    taper_starts_kph : sequence of float
        Speed at which the MGU-K taper starts with Overtake Mode, km/h; below
        the 355 km/h end of the taper.
    batteries : sequence of float
        Attacker state of charge, %.

    Returns
    -------
    dict
        Exact P per battery level for each energy and each taper start, and
        the 2026 values of both parameters.
    """
    scenarios = pd.DataFrame([{**DEFAULTS, "attacker_battery": float(b)} for b in batteries])
    end_kph = physics.TAPER_STANDARD_KPH[1]
    out = {"batteries": list(batteries),
           "rule_2026": {"energy_mj": physics.OVERTAKE_MODE_EXTRA_J / 1e6,
                         "taper_start_kph": physics.TAPER_OVERTAKE_MODE_KPH[0]},
           "energy": {"mj": list(energies_mj), "p": {str(b): [] for b in batteries}},
           "taper": {"start_kph": list(taper_starts_kph), "p": {str(b): [] for b in batteries}}}
    for e in energies_mj:
        # initial_conditions reads the energy from the simulator module's own name for it.
        with _patched(simulator, OVERTAKE_MODE_EXTRA_J=e * 1e6):
            for b, q in zip(batteries, simulator.probability(scenarios)):
                out["energy"]["p"][str(b)].append(float(q))
    for start in taper_starts_kph:
        with _patched(physics, TAPER_OVERTAKE_MODE_KPH=(float(start), end_kph)):
            for b, q in zip(batteries, simulator.probability(scenarios)):
                out["taper"]["p"][str(b)].append(float(q))
    plots.rules(out, FIGURES / "rules.png")
    return _write("rules", out)


# --- 5. Model selection ------------------------------------------------------

ARCHITECTURES = [(32,), (64, 64), (128, 128, 64), (256, 256, 128)]


def _inference_seconds(predictor, df, repeats=3):
    """Best wall-clock seconds for every ensemble member's forward pass on ``df`` (preprocessing excluded)."""
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
        """Validation metrics, default-scenario error and worst battery-axis error for one set of predictions."""
        m = metrics(y, p_val, p_true)
        return {"val_log_loss": m["log_loss"], "val_mae_vs_true": m["mae_vs_true"],
                "default_scenario_error": float(p_default - default_truth),
                "battery_axis_max_error": float(np.abs(p_axis - axis_truth).max())}

    out = {"hidden": list(HIDDEN), "seeds": list(seeds), "default_scenario_truth": default_truth}
    for name, energy_cover in (("raw inputs only", False), ("with energy-cover inputs", True)):
        runs, preds = [], []
        for seed in seeds:
            predictor, history = train_network(train_df, val_df, hidden=HIDDEN, seed=seed, physics_features=energy_cover,
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


# --- 6. Comparison with alternatives ----------------------------------------

def tune_baselines(learning_rates=(0.05, 0.1, 0.2), leaves=(15, 31, 63), cs=(0.1, 1.0, 10.0)):
    """Check that the baselines were not handicapped by their settings.

    Gradient boosting is refitted over a grid of learning rate and tree size
    (up to 1,000 trees, early stopping on an internal 10 % split) and
    logistic regression over its regularisation strength ``C``, all on the
    training set and scored on the validation set. The test set is not used.
    The settings used in `comparison` and the network's validation scores are
    recorded alongside.

    Parameters
    ----------
    learning_rates, leaves : sequence
        Gradient-boosting grid: learning rate and maximum leaves per tree.
    cs : sequence of float
        Inverse L2 regularisation strengths for logistic regression.

    Returns
    -------
    dict
        One row per setting with validation log loss and error against the
        exact probability.
    """
    train_df, val_df, _ = split(dataset(), SEED)
    y, p_true = val_df[LABEL].to_numpy(), val_df["p_true"].to_numpy()

    def score(model):
        """Fit time and validation log loss and error against the exact probability of a fitted baseline."""
        m = metrics(y, model.predict(val_df), p_true)
        return {"fit_seconds": model.fit_seconds, "val_log_loss": m["log_loss"], "val_mae_vs_true": m["mae_vs_true"]}

    out = {"gradient_boosting": [], "logistic_regression": []}
    for rate in learning_rates:
        for leaf in leaves:
            model = baselines.gradient_boosting(SEED, learning_rate=rate, max_leaf_nodes=leaf, max_iter=1000)
            model.fit(train_df)
            out["gradient_boosting"].append({"learning_rate": rate, "max_leaf_nodes": leaf, "max_iter": 1000,
                                             "trees": int(model.pipeline[-1].n_iter_), **score(model)})
            print(out["gradient_boosting"][-1], flush=True)
    for c in cs:
        model = baselines.logistic_regression(C=c).fit(train_df)
        out["logistic_regression"].append({"C": c, **score(model)})
    used = baselines.gradient_boosting(SEED).fit(train_df)
    out["used"] = {"gradient_boosting": {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 500,
                                         "trees": int(used.pipeline[-1].n_iter_), **score(used)},
                   "logistic_regression": {"C": 1.0, **score(baselines.logistic_regression().fit(train_df))}}
    network = OvertakePredictor.load()
    m = metrics(y, network.predict(val_df), p_true)
    out["network_validation"] = {"val_log_loss": m["log_loss"], "val_mae_vs_true": m["mae_vs_true"]}
    return _write("baselines", out)


def _bootstrap_interval(values, n_resamples=1000, seed=0):
    """95 % percentile bootstrap interval for the mean of ``values``."""
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_resamples)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


# Changes a strategist compares (the intended use), for the paired what-if check in `comparison`.
WHAT_IF_CHANGES = {
    "Attacker Boost switched": lambda d: d.assign(attacker_boost=~d["attacker_boost"].astype(bool)),
    "Attacker battery 20 points lower": lambda d: d.assign(attacker_battery=np.clip(d["attacker_battery"] - 20, 0, 100)),
}


def tyre_mismatch(df):
    """True where either car's tyres do not suit the surface (slicks when damp/wet, treaded tyres when dry)."""
    slicks = ["soft", "medium", "hard"]
    dry = df["track_condition"] == "dry"
    wrong = [(dry & ~df[c].isin(slicks)) | (~dry & df[c].isin(slicks)) for c in ("attacker_tyre", "defender_tyre")]
    return (wrong[0] | wrong[1]).to_numpy()


def _region_errors(test_df, p):
    """Mean absolute error against ``p_true`` per weather, per segment and with or without suited tyres."""
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
    """Quantiles (median to maximum) of the absolute error against ``p_true``, keyed ``q50`` ... ``q100``."""
    err = np.abs(p - test_df["p_true"].to_numpy())
    return {f"q{int(q * 1000) / 10:g}": float(np.quantile(err, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}


def _model_bytes(obj):
    """Memory of a model: float bytes of the network weights, or the pickled size of a scikit-learn pipeline."""
    if isinstance(obj, OvertakePredictor):
        return sum(p.numel() * p.element_size() for m in obj.models for p in m.parameters())
    return len(pickle.dumps(obj.pipeline))


def comparison():
    """Compare the network with the alternatives on the held-out test set.

    Besides the metrics of `overtake.train.metrics` for each model, records:
    a bootstrap 95 % interval of each mean error against the exact
    probability; the metrics of the exact probability itself, the floor any
    model could reach on these 0/1 outcomes; the error in *changes* of
    probability for the what-if comparisons a strategist makes
    (`WHAT_IF_CHANGES`); and extrapolation to straights of 1300-2000 m,
    longer than any in training, where a data-driven model has no evidence.
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
    out = {"test_attempts": len(test_df), "dataset_overtake_rate": float(dataset()[LABEL].mean()),
           "tyre_mismatch_share": float(tyre_mismatch(test_df).mean()),
           "models": {}, "regions": {}, "error_quantiles": {}}
    for name, model in models.items():
        p = model.predict(test_df)
        seconds, _ = _timed(lambda: model.predict(sample))
        out["models"][name] = {**metrics(y, p, p_true), "fit_seconds": fit_seconds[name],
                               "predict_us_per_scenario": 1e6 * seconds / len(sample),
                               "model_bytes": _model_bytes(model),
                               "mae_vs_true_ci95": _bootstrap_interval(np.abs(p - p_true))}
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
                                          "note": "the ground truth itself; its few outcome constants could be "
                                                  "fitted to aggregate race data, but it cannot learn from "
                                                  "labelled individual attempts"}
    out["exact_probability_reference"] = metrics(y, p_true)

    pairs = test_df.iloc[:5000]
    base = {name: model.predict(pairs) for name, model in models.items()}
    exact_base = simulator.probability(pairs)
    out["paired_differences"] = {}
    for label, change in WHAT_IF_CHANGES.items():
        changed = change(pairs)
        exact_change = simulator.probability(changed) - exact_base
        out["paired_differences"][label] = {
            "scenarios": len(pairs), "mean_abs_exact_change": float(np.abs(exact_change).mean()),
            "mean_abs_error_of_change": {name: float(np.abs(model.predict(changed) - base[name] - exact_change).mean())
                                         for name, model in models.items()}}

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


# --- 7. Performance ----------------------------------------------------------

# Stages of dataset generation whose cumulative time `performance` reports.
PROFILED_STAGES = ("sample_scenarios", "closing_run", "outcome_parameters", "draw_outcomes", "overtake_integral")


def performance():
    """Where the time and memory go, and what vectorisation and chunking buy.

    * The machine and library versions the times were measured with. Times
      are wall-clock (``time.perf_counter``), the best of three runs, or a
      single run where one run takes many seconds.
    * cProfile of generating 20,000 attempts: the functions with the most
      time of their own, and the cumulative time of each stage.
    * Vectorised integration of 200 scenarios together against one at a time.
    * Chunk size for dataset generation (array size vs cache and overhead),
      and the peak memory of integrating one 20,000-scenario chunk
      (``tracemalloc``).
    * FLOPs per prediction and achieved FLOP rate for the network (batches of
      1,024 and 10,000) and the direct simulation; arithmetic intensity of
      network inference, and an estimate for the simulator.
    * What a full training run costs: simulating the 300,000 attempts
      against training the five networks.

    Returns
    -------
    dict
    """
    import os
    import platform
    import tracemalloc

    out = {"environment": {
        "processor": platform.processor(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "timing": "wall-clock time.perf_counter; best of 3 runs, or 1 run for the loop, chunk-size, "
                  "simulation and data-generation timings"}}
    profiler = cProfile.Profile()
    profiler.enable()
    simulator.generate_dataset(20_000, seed=1)
    profiler.disable()
    stats = pstats.Stats(profiler, stream=io.StringIO()).sort_stats("tottime")
    top = sorted(stats.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:8]
    total = sum(v[2] for v in stats.stats.values())
    out["profile_top_functions"] = [
        {"function": f"{fn[0].replace(chr(92), '/').split('/')[-1]}:{fn[2]}", "calls": v[1], "own_seconds": v[2],
         "share": v[2] / total} for fn, v in top
    ]
    # Cumulative time includes everything a function calls, so the stages show where
    # the whole run goes (they overlap: closing_run contains the integration).
    out["profile_stage_shares"] = {
        fn[2]: v[3] / total for fn, v in stats.stats.items()
        if fn[2] in PROFILED_STAGES and fn[0].replace("\\", "/").endswith("overtake/simulator.py")}

    df = simulator.sample_scenarios(200, np.random.default_rng(4))
    t_vec, _ = _timed(lambda: simulator.closing_run(df))
    t_loop, _ = _timed(lambda: [simulator.closing_run(df.iloc[i:i + 1]) for i in range(len(df))], repeats=1)
    out["vectorisation_200_scenarios"] = {"loop_seconds": t_loop, "vectorised_seconds": t_vec,
                                          "speedup": t_loop / t_vec}

    out["chunk_size_40k_scenarios"] = []
    for chunk in (1_000, 5_000, 20_000, 40_000):
        seconds, _ = _timed(lambda: simulator.generate_dataset(40_000, seed=2, chunk=chunk), repeats=1)
        out["chunk_size_40k_scenarios"].append({"chunk": chunk, "seconds": seconds})
    chunked = min(row["seconds"] for row in out["chunk_size_40k_scenarios"]) / 40_000
    out["loop_to_final_design"] = {"loop_us_per_scenario": 1e6 * t_loop / len(df),
                                   "chunked_us_per_scenario": 1e6 * chunked,
                                   "speedup": t_loop / len(df) / chunked}

    # NumPy reports its array allocations to tracemalloc, so the peak covers the
    # RK4 stages and temporaries, not only the six state rows.
    chunk_df = simulator.sample_scenarios(20_000, np.random.default_rng(6))
    tracemalloc.start()
    simulator.closing_run(chunk_df)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sample = simulator.sample_scenarios(10_000, np.random.default_rng(5))
    t_sim, _ = _timed(lambda: simulator.probability(sample), repeats=1)
    run = simulator.closing_run(sample)[0]
    steps = run.time_s / simulator.TIME_STEP_S
    sim_flops = float(np.mean(steps) * FLOPS_PER_RK4_STEP + FLOPS_QUADRATURE)
    network = OvertakePredictor.load()
    hidden = network.model.hidden
    t_nn = _inference_seconds(network, sample)
    batches = [sample.iloc[i:i + 1024] for i in range(0, len(sample), 1024)]
    t_nn_1024 = sum(_inference_seconds(network, b) for b in batches)
    nn = nn_flops(hidden) * len(network.models)
    params = network.n_parameters
    width_in = network.model.mlp[0].in_features
    sim_gflops = sim_flops * len(sample) / t_sim / 1e9
    out["flops"] = {
        "network_hidden": list(hidden),
        "network_members": len(network.models),
        "network_flops_per_prediction": nn,
        "network_gflops_achieved": nn * len(sample) / t_nn / 1e9,
        "network_gflops_achieved_batch_1024": nn * len(sample) / t_nn_1024 / 1e9,
        "network_us_per_prediction_batch_10k": 1e6 * t_nn / len(sample),
        "simulation_mean_rk4_steps": float(np.mean(steps)),
        "simulation_flops_per_scenario_estimate": sim_flops,
        "simulation_gflops_achieved": sim_gflops,
        "simulation_us_per_scenario": 1e6 * t_sim / len(sample),
        # Weights (4 bytes each) are read once per batch; each prediction reads
        # its inputs and writes one output.
        "network_arithmetic_intensity_flop_per_byte": {
            str(b): nn * b / (4 * params + 4 * b * (width_in + 1)) for b in (1, 64, 1024, 10_000)},
        # Each element-wise float64 operation reads two 8-byte operands and writes
        # one result for a single FLOP, about 1/24 FLOP/byte (an estimate: some
        # operands stay in cache), so the achieved rate implies the memory traffic.
        "simulation_arithmetic_intensity_flop_per_byte_estimate": 1 / 24,
        "simulation_implied_bandwidth_gb_per_s": 24 * sim_gflops,
    }
    out["memory"] = {
        "network_parameter_bytes": 4 * params,
        "dataset_300k_bytes": int(dataset().memory_usage(deep=True).sum()),
        "ode_state_bytes_per_20k_chunk": 6 * 8 * 20_000,
        "simulation_peak_bytes_per_20k_chunk": int(peak),
    }
    t_data, _ = _timed(lambda: simulator.generate_dataset(300_000, seed=SEED), repeats=1)
    checkpoint = torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=True)
    # One epoch of one network, timed here, so data generation and training compare on one machine.
    train_df, val_df, _ = split(dataset(), SEED)
    t_epoch, _ = _timed(lambda: train_network(train_df, val_df, epochs=1, seed=SEED, log=lambda _: None),
                        repeats=1)
    out["training_pipeline"] = {"data_generation_seconds_300k": t_data,
                                "one_epoch_one_network_seconds": t_epoch,
                                "ensemble_training_seconds": checkpoint["fit_seconds"],
                                "first_network_epochs": len(checkpoint["history"]),
                                "note": "ensemble_training_seconds was measured when the model was trained "
                                        "(python -m overtake.train, another machine); the rest is timed here"}
    return _write("performance", out)


# --- 8. Sensitivity ----------------------------------------------------------

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

    # Along the attacker-battery axis through the default scenario: where is the network furthest off?
    dry = pd.DataFrame([{**DEFAULTS, "attacker_battery": b} for b in np.arange(30.0, 101.0, 2.5)])
    dry_err = network.predict(dry) - simulator.probability(dry)
    worst = int(np.argmax(np.abs(dry_err)))
    dry_axis = {"worst_error": float(dry_err[worst]), "worst_at_battery": float(dry["attacker_battery"][worst]),
                "error_at_full_battery": float(dry_err[-1])}

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
                                  "default_battery_axis": dry_axis,
                                  "wet_intermediates_battery_axis": wet_axis,
                                  "test_wet_straights_treaded_tyres": wet_test})


# --- 9. Real 2026 races -------------------------------------------------------

def real():
    """Compare the simulator with real 2026 races from OpenF1 (`overtake.realdata`).

    Downloads and caches the data on first run (about 4 minutes), then works
    offline. Writes ``reports/results/real_data.json`` and four figures.
    """
    from . import realdata
    realdata.download()  # downloads once (about 4 minutes), then reads the cache
    result = realdata.analyse()
    plots.real_telemetry(result["telemetry"], FIGURES / "real_telemetry.png")
    plots.real_gap(result["gap"], FIGURES / "real_gap.png")
    plots.real_circuits(result, FIGURES / "real_circuits.png")
    plots.real_speed_traps(result["speed_traps"], FIGURES / "real_speed_traps.png")
    return _write("real_data", result)


# --- 10. Workflow diagram ------------------------------------------------------

def diagram():
    """Draw the workflow overview used in the documentation (no computation)."""
    plots.workflow_diagram(FIGURES / "workflow.png")
    return {"figure": str(FIGURES / "workflow.png")}


def figures():
    """Redraw every figure that can be drawn from the saved results alone.

    Lets the figures' style change (fonts, colours, labels) without rerunning
    the experiments, several of which train networks for many minutes. The
    figures that need a model or simulated traces (calibration, what-if,
    speed trace) are redrawn by `comparison`, `sensitivity` and
    `speed_trace`. Missing result files are skipped.

    Returns
    -------
    dict
        ``redrawn`` and ``skipped`` figure file names.
    """
    def load(name):
        """The saved ``reports/results/<name>.json`` as a dict, or None if that experiment has not been run."""
        path = RESULTS / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    drawers = {
        "solver_convergence.png": ("solver", lambda r, path: plots.convergence(r, path)),
        "timestep_study.png": ("timestep", lambda r, path: plots.timestep(r, path, simulator.TIME_STEP_S)),
        "stability.png": ("stability", lambda r, path: plots.stability(r, path)),
        "quadrature_study.png": ("quadrature", lambda r, path: plots.quadrature(r, path)),
        "constants.png": ("constants", lambda r, path: plots.constants(r, path)),
        "rules.png": ("rules", lambda r, path: plots.rules(r, path)),
        "model_selection.png": ("selection",
                                lambda r, path: plots.model_selection(r["architectures"], r["data_sizes"], path)),
        "method_comparison.png": ("comparison", lambda r, path: plots.method_comparison(r["regions"], path)),
        "sensitivity.png": ("sensitivity", lambda r, path: plots.sensitivity(r["rows"], path)),
        "real_telemetry.png": ("real_data", lambda r, path: plots.real_telemetry(r["telemetry"], path)),
        "real_gap.png": ("real_data", lambda r, path: plots.real_gap(r["gap"], path)),
        "real_circuits.png": ("real_data", lambda r, path: plots.real_circuits(r, path)),
        "real_speed_traps.png": ("real_data", lambda r, path: plots.real_speed_traps(r["speed_traps"], path)),
    }
    out = {"redrawn": [], "skipped": []}
    for figure, (name, draw) in drawers.items():
        result = load(name)
        if result is None:
            out["skipped"].append(figure)
            continue
        draw(result, FIGURES / figure)
        out["redrawn"].append(figure)
    plots.workflow_diagram(FIGURES / "workflow.png")
    out["redrawn"].append("workflow.png")
    return out


EXPERIMENTS = {
    "solver": solver,
    "timestep": timestep,
    "stability": stability,
    "adaptive": adaptive,
    "quadrature": quadrature,
    "speed_trace": speed_trace,
    "constants": constants,
    "rules": rules,
    "selection": selection,
    "ablation": ablation,
    "baselines": tune_baselines,
    "comparison": comparison,
    "performance": performance,
    "sensitivity": sensitivity,
    "real": real,
    "diagram": diagram,
    "figures": figures,
}


def main(argv=None):
    """Run the named experiments (``all`` for every one, which is also the default).

    Parameters
    ----------
    argv : list of str, optional
        Experiment names; ``sys.argv[1:]`` if omitted.

    Raises
    ------
    SystemExit
        With status 2 if a name is not an experiment.
    """
    names = (argv if argv is not None else sys.argv[1:]) or ["all"]
    if names == ["all"]:
        # `figures` only redraws what the experiments have just drawn.
        names = [name for name in EXPERIMENTS if name != "figures"]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s) {unknown}; choose from {list(EXPERIMENTS)} or 'all'", file=sys.stderr)
        raise SystemExit(2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names:
        start = time.perf_counter()
        print(f"== {name}", flush=True)
        result = EXPERIMENTS[name]()
        print(json.dumps(result, indent=1, default=float)[:3000], flush=True)
        print(f"== {name} done in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
