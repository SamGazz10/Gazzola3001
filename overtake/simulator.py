"""Simulator of one overtaking attempt under the 2026 rules.

No public dataset of 2026 overtakes records battery charge and Boost /
Overtake Mode use, so the network is trained on attempts generated here. An
attempt has two stages.

**1. Closing (deterministic physics).** `overtake.physics` integrates both
cars from the corner exit to the braking point and returns the gap the
attacker still has to close, ``D`` (s). Tyres set grip and corner-exit speed,
the battery and Boost / Overtake Mode set the electrical power, the slipstream
cuts the attacker's drag, and car performance differences shift exit speed and
drag. Poor-weather caution and the defender's positioning add to ``D``.

**2. Execution (random).** Driver execution varies from attempt to attempt,
modelled as Gaussian noise ``eps ~ N(0, sigma)`` on the deficit. The attacker
can lunge if ``D - eps`` is within ``LUNGE_RANGE_S``; the margin is then
``u = eps - (D - LUNGE_RANGE_S)``. A lunge sticks with probability
``logistic(z0 + 5 min(u, 0.5))``, where ``z0`` collects driver ratings, braking
grip, corner type, weather and circuit.

Because the randomness is a single Gaussian variable, the exact probability of
an overtake is a one-dimensional integral,

    P = integral_0^inf  phi_sigma(u + D - LUNGE_RANGE_S) logistic(z0 + 5 min(u, 0.5)) du,

which `probability` evaluates by Gauss-Legendre quadrature on [0, 0.5] and in
closed form beyond. This gives a ground truth to validate the network
against. `simulate` instead draws the random variables, producing the 0/1
outcomes the network is trained on, exactly as real race data would look.

Constants marked "estimate" are engineering judgements chosen to give
plausible magnitudes, not values measured from F1 data.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

from .features import (CIRCUITS, COMPOUNDS, CONDITIONS, LABEL, SEGMENTS, TRAINED_RUN_LENGTH_M,
                       TRAINED_TRACK_TEMP_C, validate)
from .physics import (BATTERY_USABLE_J, OVERTAKE_MODE_EXTRA_J, CarParameters, CarState, rk4_step,
                      run_to_braking_point, two_car_rhs)

CAR = CarParameters()
STANDARD_MAP_SHARE = 0.55       # share of MGU-K power used without Boost (estimate)
EXIT_SPEED_MPS = 33.0           # corner-exit speed onto the run, new softs, dry (estimate, ~120 km/h)
DIRTY_AIR_GRIP_LOSS = 0.06      # grip lost in the exit corner when right behind another car (estimate)
TOP_SPEED_REF_MPS = 92.0        # reference top speed for converting a speed advantage to drag
LAP_TIME_REF_S = 90.0           # reference lap time for converting a pace advantage to exit speed
TIME_STEP_S = 0.05              # ODE step (see experiments.timestep)
FULL_BATTERY_SHARE = 0.15       # share of sampled attempts starting with a full battery

LUNGE_RANGE_S = 0.15            # deficit at the braking point from which a lunge is possible (estimate)
MAX_MARGIN_S = 0.5
MARGIN_LOGIT_PER_S = 5.0

EXECUTION_SD = {"dry": 0.20, "damp": 0.24, "wet": 0.28}   # s (estimate)
CAUTION_S = {"dry": 0.0, "damp": 0.05, "wet": 0.12}       # time given away for visibility/grip (estimate)
CONDITION_LOGIT = {"dry": 0.0, "damp": -0.5, "wet": -0.9}
SEGMENT_LOGIT = {"straight": 0.0, "slow_corner": 0.2, "medium_corner": -0.6, "fast_corner": -1.6}

# Grip of each compound on each surface relative to a new soft on a dry track.
GRIP = {
    "dry": {"soft": 1.000, "medium": 0.988, "hard": 0.976, "intermediate": 0.900, "wet": 0.860},
    "damp": {"soft": 0.800, "medium": 0.790, "hard": 0.780, "intermediate": 0.860, "wet": 0.830},
    "wet": {"soft": 0.550, "medium": 0.540, "hard": 0.530, "intermediate": 0.720, "wet": 0.760},
}
DEG_PER_LAP = {"soft": 0.0018, "medium": 0.0011, "hard": 0.0007, "intermediate": 0.0015, "wet": 0.0012}
CLIFF_LAP = {"soft": 22, "medium": 35, "hard": 50, "intermediate": 25, "wet": 30}
CLIFF_DEG_PER_LAP = 0.004

# How readily a move sticks at each circuit once alongside (logit offset).
# Estimates from track width, braking zones and overtaking history.
PASSABILITY = {
    "generic": 0.0, "melbourne": -0.1, "shanghai": 0.4, "suzuka": -0.4, "bahrain": 0.6,
    "jeddah": 0.3, "miami": 0.1, "montreal": 0.4, "monaco": -2.5, "barcelona": -0.3,
    "spielberg": 0.4, "silverstone": 0.1, "spa": 0.6, "hungaroring": -1.0, "zandvoort": -1.2,
    "monza": 0.5, "madrid": 0.0, "baku": 0.5, "singapore": -0.9, "austin": 0.3,
    "mexico_city": 0.3, "interlagos": 0.4, "las_vegas": 0.5, "lusail": 0.0, "yas_marina": 0.1,
}
assert set(PASSABILITY) == set(CIRCUITS)


def _lookup(table, keys):
    return pd.Series(np.asarray(keys)).map(table).to_numpy(dtype=float)


def tyre_grip(compound, age, condition, track_temp):
    """Grip of a tyre relative to a new soft on a dry track.

    Parameters
    ----------
    compound, condition : array_like of str
        Tyre compound and track condition per scenario.
    age : array_like
        Tyre age in laps.
    track_temp : array_like
        Track temperature, deg C. Hot tracks wear tyres faster; hard tyres
        lose grip on a cold track.

    Returns
    -------
    numpy.ndarray
        Relative grip, about 0.5-1.0.
    """
    compound, condition = np.asarray(compound), np.asarray(condition)
    age, track_temp = np.asarray(age, dtype=float), np.asarray(track_temp, dtype=float)
    base = pd.Series(list(zip(condition, compound))).map(
        {(c, t): GRIP[c][t] for c in CONDITIONS for t in COMPOUNDS}
    ).to_numpy(dtype=float)
    deg = _lookup(DEG_PER_LAP, compound) * np.clip(1 + 0.02 * (track_temp - 35), 0.6, 1.6)
    # Treaded tyres overheat and wear quickly on a dry track.
    deg = np.where((condition == "dry") & np.isin(compound, ["intermediate", "wet"]), deg * 3, deg)
    wear = deg * age + CLIFF_DEG_PER_LAP * np.maximum(0, age - _lookup(CLIFF_LAP, compound))
    cold_hard = (compound == "hard") & (track_temp < 25)
    return base - wear - 0.01 * cold_hard


def initial_conditions(df, car=CAR):
    """Translate scenarios into the physical inputs of the two-car model.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema.
    car : CarParameters, optional

    Returns
    -------
    dict
        ``attacker`` and ``defender`` (`CarState`), corner-exit speeds
        ``v0_a`` and ``v0_d`` (m/s), ``gap`` (s), ``length`` (m) and relative
        tyre grip ``grip_a`` and ``grip_d``.
    """
    cond = df["track_condition"].to_numpy()
    temp = df["track_temp_c"].to_numpy(dtype=float)
    gap = df["gap_s"].to_numpy(dtype=float)
    grip_a = tyre_grip(df["attacker_tyre"], df["attacker_tyre_age"], cond, temp)
    grip_d = tyre_grip(df["defender_tyre"], df["defender_tyre_age"], cond, temp)

    om = df["overtake_mode"].to_numpy(dtype=bool)
    boost_a = df["attacker_boost"].to_numpy(dtype=bool)
    boost_d = df["defender_boost"].to_numpy(dtype=bool)
    speed_gain = df["top_speed_delta_kph"].to_numpy(dtype=float) / 3.6
    # At fixed power, top speed scales with CdA^(-1/3).
    cda_a = car.cda_m2 * (TOP_SPEED_REF_MPS / (TOP_SPEED_REF_MPS + speed_gain)) ** 3

    attacker = CarState(
        mu=car.mu_reference * grip_a,
        cda_m2=cda_a,
        deploy_share=np.where(boost_a, 1.0, STANDARD_MAP_SHARE),
        energy_j=BATTERY_USABLE_J * df["attacker_battery"].to_numpy(dtype=float) / 100 + OVERTAKE_MODE_EXTRA_J * om,
        overtake_mode=om,
    )
    defender = CarState(
        mu=car.mu_reference * grip_d,
        cda_m2=np.full(len(df), car.cda_m2),
        deploy_share=np.where(boost_d, 1.0, STANDARD_MAP_SHARE),
        energy_j=BATTERY_USABLE_J * df["defender_battery"].to_numpy(dtype=float) / 100,
    )
    # Corner speed scales with sqrt(grip); following closely costs grip (dirty
    # air); a lap-time advantage is applied as a proportional speed advantage.
    v0_d = EXIT_SPEED_MPS * np.sqrt(grip_d)
    v0_a = (EXIT_SPEED_MPS * np.sqrt(grip_a * (1 - DIRTY_AIR_GRIP_LOSS * np.exp(-gap)))
            * (1 + df["pace_delta_s"].to_numpy(dtype=float) / LAP_TIME_REF_S))
    return {"attacker": attacker, "defender": defender, "v0_a": v0_a, "v0_d": v0_d, "gap": gap,
            "length": df["run_length_m"].to_numpy(dtype=float), "grip_a": grip_a, "grip_d": grip_d}


def closing_run(df, dt=TIME_STEP_S, method="rk4", car=CAR, crossing="hermite"):
    """Stage 1: integrate both cars along the run for every scenario.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema.
    dt : float, optional
        Integration time step, s.
    method : {"rk4", "euler"}, optional
    car : CarParameters, optional
    crossing : {"hermite", "linear"}, optional
        Interpolation used to locate the braking point within the last step
        (see `overtake.physics.run_to_braking_point`).

    Returns
    -------
    run : overtake.physics.RunResult
        State at the braking point, including the deficit in seconds.
    grip_a, grip_d : numpy.ndarray
        Relative tyre grip of the attacker and defender.
    """
    ic = initial_conditions(df, car)
    run = run_to_braking_point(ic["length"], ic["gap"], ic["v0_a"], ic["v0_d"], ic["attacker"], ic["defender"],
                               car, dt=dt, method=method, crossing=crossing)
    return run, ic["grip_a"], ic["grip_d"]


def speed_trace(scenario, dt=TIME_STEP_S, car=CAR):
    """Time history of both cars along the run for one scenario.

    Parameters
    ----------
    scenario : dict
        A complete scenario in the `overtake.features` schema.
    dt : float, optional
    car : CarParameters, optional

    Returns
    -------
    pandas.DataFrame
        Columns ``t`` (s), ``attacker_x``, ``defender_x`` (m, from the corner
        exit), ``attacker_kph``, ``defender_kph`` and ``attacker_mj``,
        ``defender_mj`` (electrical energy used), one row per step until the
        defender reaches the braking point.

    Raises
    ------
    ValueError
        If the scenario does not match the schema (`overtake.features.validate`).
    """
    df = pd.DataFrame([scenario])
    validate(df)
    ic = initial_conditions(df, car)
    y = np.array([[-ic["gap"][0] * ic["v0_d"][0]], [ic["v0_a"][0]], [0.0], [0.0], [ic["v0_d"][0]], [0.0]])
    rows, t = [], 0.0
    while y[3, 0] < ic["length"][0]:
        rows.append([t, *y[:, 0]])
        y = rk4_step(lambda s: two_car_rhs(s, ic["attacker"], ic["defender"], car), y, dt)
        t += dt
    trace = pd.DataFrame(rows, columns=["t", "attacker_x", "attacker_v", "attacker_e",
                                        "defender_x", "defender_v", "defender_e"])
    for who in ("attacker", "defender"):
        trace[f"{who}_kph"] = trace.pop(f"{who}_v") * 3.6
        trace[f"{who}_mj"] = trace.pop(f"{who}_e") / 1e6
    return trace


def outcome_parameters(df, run, grip_a, grip_d):
    """Stage 2 inputs: effective deficit, execution noise and completion logit.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios.
    run : overtake.physics.RunResult
        Output of `closing_run`.
    grip_a, grip_d : numpy.ndarray
        Relative tyre grip.

    Returns
    -------
    deficit : numpy.ndarray
        Deficit at the braking point including caution and defensive
        positioning, s.
    sigma : numpy.ndarray
        Standard deviation of execution noise, s.
    z0 : numpy.ndarray
        Completion logit with zero margin.
    """
    cond = df["track_condition"].to_numpy()
    skill_a = df["attacker_skill"].to_numpy(dtype=float)
    skill_d = df["defender_skill"].to_numpy(dtype=float)
    deficit = run.deficit_s + _lookup(CAUTION_S, cond) + 0.04 * (skill_d - 60) / 40
    sigma = _lookup(EXECUTION_SD, cond) + 0.04 * np.clip((90 - skill_a) / 30, 0, 1)
    z0 = (
        0.5
        + 1.2 * (skill_a - skill_d) / 40
        + 25.0 * (grip_a - grip_d)
        + _lookup(SEGMENT_LOGIT, df["segment"])
        + _lookup(CONDITION_LOGIT, cond)
        + _lookup(PASSABILITY, df["circuit"])
        # A defender with energy left can fight back on the exit.
        - 0.2 * df["defender_boost"].to_numpy(dtype=float) * df["defender_battery"].to_numpy(dtype=float) / 100
    )
    return deficit, sigma, z0


def overtake_integral(deficit, sigma, z0, rule="gauss", n=16):
    """Probability of an overtake given the stage-2 parameters.

    Evaluates ``P = int_0^inf phi_sigma(u + D - L) logistic(z0 + k min(u, m)) du``
    (``L`` = lunge range, ``k`` = margin logit slope, ``m`` = margin cap). The
    tail beyond ``u = m`` has a closed form; the integral over ``[0, m]`` is
    computed numerically.

    Parameters
    ----------
    deficit, sigma, z0 : numpy.ndarray
        From `outcome_parameters`.
    rule : {"gauss", "simpson", "trapezoid"}, optional
        Quadrature rule for ``[0, m]``: Gauss-Legendre with ``n`` nodes, or
        composite Simpson / trapezoid with ``n`` intervals (``n`` even for
        Simpson).
    n : int, optional

    Returns
    -------
    numpy.ndarray
        Probability in [0, 1].
    """
    deficit, sigma, z0 = (np.asarray(a, dtype=float)[..., None] for a in (deficit, sigma, z0))
    shift = deficit - LUNGE_RANGE_S

    if rule == "gauss":
        nodes, weights = np.polynomial.legendre.leggauss(n)
        u = 0.5 * MAX_MARGIN_S * (nodes + 1)
        w = 0.5 * MAX_MARGIN_S * weights
    else:
        u = np.linspace(0, MAX_MARGIN_S, n + 1)
        h = MAX_MARGIN_S / n
        if rule == "trapezoid":
            w = np.full(n + 1, h)
            w[[0, -1]] = h / 2
        elif rule == "simpson":
            if n % 2:
                raise ValueError("Simpson's rule needs an even number of intervals")
            w = np.where(np.arange(n + 1) % 2, 4.0, 2.0) * h / 3
            w[[0, -1]] = h / 3
        else:
            raise ValueError(f"unknown rule {rule!r}")

    density = np.exp(-0.5 * ((u + shift) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    body = (density * expit(z0 + MARGIN_LOGIT_PER_S * u) * w).sum(axis=-1)
    tail = expit(z0 + MARGIN_LOGIT_PER_S * MAX_MARGIN_S) * (1 - ndtr((MAX_MARGIN_S + shift) / sigma))
    return body + tail[..., 0]


def probability(df, dt=TIME_STEP_S, car=CAR):
    """Exact probability of an overtake for each scenario (physics + quadrature).

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema.
    dt : float, optional
        ODE time step, s.
    car : CarParameters, optional
        Physical parameters of both cars (varied by
        ``overtake.experiments.constants``).

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If ``df`` does not match the schema (`overtake.features.validate`):
        without this check an unknown category would give NaN silently.
    """
    validate(df)
    run, grip_a, grip_d = closing_run(df, dt=dt, car=car)
    return overtake_integral(*outcome_parameters(df, run, grip_a, grip_d))


def draw_outcomes(deficit, sigma, z0, rng):
    """Play out one attempt per scenario by drawing the random variables.

    Parameters
    ----------
    deficit, sigma, z0 : numpy.ndarray
        From `outcome_parameters`.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray of bool
        True where the overtake stuck.
    """
    eps = rng.normal(0.0, sigma)
    margin = eps - (deficit - LUNGE_RANGE_S)
    sticks = rng.random(margin.shape) < expit(z0 + MARGIN_LOGIT_PER_S * np.minimum(margin, MAX_MARGIN_S))
    return (margin >= 0) & sticks


def monte_carlo_probability(df, n_runs, seed=1):
    """Estimate each scenario's probability by replaying it ``n_runs`` times.

    Used to check `probability`; its standard error is ``sqrt(p (1-p) / n_runs)``.
    """
    rng = np.random.default_rng(seed)
    params = outcome_parameters(df, *closing_run(df))
    repeated = [np.repeat(a[:, None], n_runs, axis=1) for a in params]
    return draw_outcomes(*repeated, rng).mean(axis=1)


# Spread of the gap at the Overtake Mode detection point around the gap when
# the attack starts (s): the two are measured at different places on the lap.
DETECTION_GAP_SD_S = 0.2


def sample_scenarios(n, rng):
    """Draw ``n`` varied race situations covering the whole input space.

    Parameters
    ----------
    n : int
    rng : numpy.random.Generator

    Returns
    -------
    pandas.DataFrame
        Scenarios in the `overtake.features` schema (no label).
    """
    cond = rng.choice(CONDITIONS, n, p=[0.75, 0.13, 0.12])
    tyre_probs = {
        "dry": [0.30, 0.38, 0.26, 0.04, 0.02],
        "damp": [0.10, 0.10, 0.08, 0.57, 0.15],
        "wet": [0.03, 0.03, 0.04, 0.48, 0.42],
    }

    def tyres():
        out = np.empty(n, dtype=object)
        for c in CONDITIONS:
            mask = cond == c
            out[mask] = rng.choice(COMPOUNDS, mask.sum(), p=tyre_probs[c])
        return out

    def battery():
        # A full battery is a common race state (after a safety car or a
        # harvesting lap), so 15 % of attempts start at exactly 100 %. Without
        # this, 100 % sits at the edge of the training data, where the network
        # was found to be least accurate (experiments.sensitivity).
        return np.where(rng.random(n) < FULL_BATTERY_SHARE, 100.0, rng.uniform(0, 100, n))

    seg = rng.choice(SEGMENTS, n, p=[0.55, 0.20, 0.15, 0.10])
    gap = rng.uniform(0.05, 2.0, n)
    # The gap at the detection point differs a little from the gap when the attack starts.
    detection_gap = gap + rng.normal(0.0, DETECTION_GAP_SD_S, n)
    return pd.DataFrame({
        "track_condition": cond,
        "segment": seg,
        "circuit": rng.choice(CIRCUITS, n),
        "attacker_tyre": tyres(),
        "defender_tyre": tyres(),
        "gap_s": gap,
        "run_length_m": np.where(seg == "straight", rng.uniform(*TRAINED_RUN_LENGTH_M["straight"], n),
                                 rng.uniform(*TRAINED_RUN_LENGTH_M["corner"], n)),
        "attacker_battery": battery(),
        "defender_battery": battery(),
        "attacker_tyre_age": rng.integers(0, 41, n).astype(float),
        "defender_tyre_age": rng.integers(0, 41, n).astype(float),
        "track_temp_c": np.where(cond == "dry", rng.uniform(*TRAINED_TRACK_TEMP_C["dry"], n),
                                 rng.uniform(*TRAINED_TRACK_TEMP_C["wet"], n)),
        "attacker_skill": rng.uniform(60, 100, n),
        "defender_skill": rng.uniform(60, 100, n),
        "top_speed_delta_kph": np.clip(rng.normal(0, 5, n), -15, 15),
        "pace_delta_s": np.clip(rng.normal(0, 0.4, n), -1.5, 1.5),
        "attacker_boost": rng.random(n) < 0.8,
        "defender_boost": rng.random(n) < 0.5,
        "overtake_mode": detection_gap <= 1.0,
    })


def generate_dataset(n, seed=0, chunk=20_000):
    """Sample ``n`` scenarios and simulate one attempt each.

    Parameters
    ----------
    n : int
        Number of attempts.
    seed : int, optional
    chunk : int, optional
        Scenarios integrated together; bounds memory and keeps arrays
        cache-friendly (see ``experiments.performance``).

    Returns
    -------
    pandas.DataFrame
        Scenarios plus ``overtake`` (0/1 outcome) and ``p_true`` (exact
        probability, for validation only - never a model input).
    """
    rng = np.random.default_rng(seed)
    df = sample_scenarios(n, rng)
    labels, p_true = np.empty(n, dtype=int), np.empty(n)
    for start in range(0, n, chunk):
        part = df.iloc[start:start + chunk]
        params = outcome_parameters(part, *closing_run(part))
        labels[start:start + chunk] = draw_outcomes(*params, rng)
        p_true[start:start + chunk] = overtake_integral(*params)
    df[LABEL] = labels
    df["p_true"] = p_true
    return df
