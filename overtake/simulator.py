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
grip, corner type, weather, circuit and the defender's remaining Boost energy.

Because the randomness is a single Gaussian variable, the exact probability of
an overtake is a one-dimensional integral,

    P = integral_0^inf  phi_sigma(u + D - LUNGE_RANGE_S) logistic(z0 + 5 min(u, 0.5)) du,

which `probability` evaluates by Gauss-Legendre quadrature on [0, 0.5] and in
closed form beyond. This gives a ground truth to validate the network
against. `draw_outcomes` (called by `generate_dataset`) instead draws the
random variables, producing the 0/1 outcomes the network is trained on,
exactly as real race data would look.

`speed_trace` records the time history of one attempt for plots, and
`monte_carlo_probability` checks `probability` by replaying attempts.

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

CAR = CarParameters()           # default physical parameters, shared by both cars (overtake.physics)
STANDARD_MAP_SHARE = 0.55       # share of MGU-K power used without Boost (estimate)
EXIT_SPEED_MPS = 33.0           # corner-exit speed onto the run, new softs, dry (estimate, ~120 km/h)
DIRTY_AIR_GRIP_LOSS = 0.06      # grip lost in the exit corner when right behind another car (estimate)
TOP_SPEED_REF_MPS = 92.0        # reference top speed (~331 km/h) for converting a speed advantage to drag (estimate)
LAP_TIME_REF_S = 90.0           # reference lap time for converting a pace advantage to exit speed (estimate)
# RK4 step: max deficit error 0.06 ms against a 0.001 s reference (experiments.timestep); predicted
# stability limit 0.16 s with Boost, first energy overshoot at 0.125 s (experiments.stability).
TIME_STEP_S = 0.05
FULL_BATTERY_SHARE = 0.15       # share of sampled attempts starting with a full battery (data design)

LUNGE_RANGE_S = 0.15            # deficit at the braking point from which a lunge is possible (estimate)
MAX_MARGIN_S = 0.5              # margin (s) beyond which a bigger lead no longer helps the move stick (estimate)
MARGIN_LOGIT_PER_S = 5.0        # log-odds gained per second of margin; the full 0.5 s adds 2.5 (estimate)

EXECUTION_SD = {"dry": 0.20, "damp": 0.24, "wet": 0.28}   # s.d. of the execution noise eps, s (estimate)
CAUTION_S = {"dry": 0.0, "damp": 0.05, "wet": 0.12}       # time given away for visibility/grip (estimate)
CONDITION_LOGIT = {"dry": 0.0, "damp": -0.5, "wet": -0.9}  # log-odds offset for completing a move (estimate)
# Log-odds offset by where the move is completed; hardest in a fast corner (estimate).
SEGMENT_LOGIT = {"straight": 0.0, "slow_corner": 0.2, "medium_corner": -0.6, "fast_corner": -1.6}

# Grip of each compound on each surface relative to a new soft on a dry track (estimates).
GRIP = {
    "dry": {"soft": 1.000, "medium": 0.988, "hard": 0.976, "intermediate": 0.900, "wet": 0.860},
    "damp": {"soft": 0.800, "medium": 0.790, "hard": 0.780, "intermediate": 0.860, "wet": 0.830},
    "wet": {"soft": 0.550, "medium": 0.540, "hard": 0.530, "intermediate": 0.720, "wet": 0.760},
}
# Relative grip lost per lap on a 35 C track; tyre_grip scales it with temperature (estimate).
DEG_PER_LAP = {"soft": 0.0018, "medium": 0.0011, "hard": 0.0007, "intermediate": 0.0015, "wet": 0.0012}
# Tyre age (laps) beyond which wear speeds up (estimate).
CLIFF_LAP = {"soft": 22, "medium": 35, "hard": 50, "intermediate": 25, "wet": 30}
CLIFF_DEG_PER_LAP = 0.004       # extra grip lost per lap beyond CLIFF_LAP (estimate)

# How readily a move sticks at each circuit once alongside (logit offset).
# Estimates from track width, braking zones and overtaking history.
PASSABILITY = {
    "generic": 0.0, "melbourne": -0.1, "shanghai": 0.4, "suzuka": -0.4, "bahrain": 0.6,
    "jeddah": 0.3, "miami": 0.1, "montreal": 0.4, "monaco": -2.5, "barcelona": -0.3,
    "spielberg": 0.4, "silverstone": 0.1, "spa": 0.6, "hungaroring": -1.0, "zandvoort": -1.2,
    "monza": 0.5, "madrid": 0.0, "baku": 0.5, "singapore": -0.9, "austin": 0.3,
    "mexico_city": 0.3, "interlagos": 0.4, "las_vegas": 0.5, "lusail": 0.0, "yas_marina": 0.1,
}
# Every circuit the schema accepts needs an offset, or _lookup would return NaN for it.
assert set(PASSABILITY) == set(CIRCUITS)


def _lookup(table, keys):
    """Map each key in ``keys`` to ``table[key]`` as a float array (vectorised).

    An unknown key gives NaN without an error, so inputs must pass `overtake.features.validate` first.
    """
    return pd.Series(np.asarray(keys)).map(table).to_numpy(dtype=float)


def tyre_grip(compound, age, condition, track_temp):
    """Grip of a tyre relative to a new soft on a dry track.

    Parameters
    ----------
    compound, condition : array_like of str
        Tyre compound (one of `overtake.features.COMPOUNDS`) and track
        condition (one of `overtake.features.CONDITIONS`), one per scenario.
    age : array_like
        Tyre age in laps.
    track_temp : array_like
        Track temperature, deg C. Hot tracks wear tyres faster; hard tyres
        lose grip on a track below 25 C.

    Returns
    -------
    numpy.ndarray of float
        Relative grip per scenario: 1.0 for a new soft on a dry track, about
        0.41-1.0 over the training ranges, and down to about 0.09 at the
        accepted limits (80-lap soft on a wet track at 65 C or hotter). It is
        positive for every input `overtake.features.validate` accepts. An
        unknown compound or condition gives NaN (inputs are not validated here).
    """
    compound, condition = np.asarray(compound), np.asarray(condition)
    age, track_temp = np.asarray(age, dtype=float), np.asarray(track_temp, dtype=float)
    # GRIP[condition][compound] for every scenario at once, via a flattened (condition, compound) -> grip map.
    base = pd.Series(list(zip(condition, compound))).map(
        {(c, t): GRIP[c][t] for c in CONDITIONS for t in COMPOUNDS}
    ).to_numpy(dtype=float)
    # Wear rate changes by 2 % per deg C about a 35 C track (estimate), limited to 0.6-1.6 times the
    # nominal rate; the limits bind below 15 C and above 65 C.
    deg = _lookup(DEG_PER_LAP, compound) * np.clip(1 + 0.02 * (track_temp - 35), 0.6, 1.6)
    # Treaded tyres overheat and wear three times as fast on a dry track (estimate).
    deg = np.where((condition == "dry") & np.isin(compound, ["intermediate", "wet"]), deg * 3, deg)
    # Linear wear, plus CLIFF_DEG_PER_LAP per lap beyond the compound's cliff lap.
    wear = deg * age + CLIFF_DEG_PER_LAP * np.maximum(0, age - _lookup(CLIFF_LAP, compound))
    # A hard tyre on a track below 25 C stays under its working temperature: -0.01 grip (estimate).
    cold_hard = (compound == "hard") & (track_temp < 25)
    return base - wear - 0.01 * cold_hard


def initial_conditions(df, car=CAR):
    """Translate scenarios into the physical inputs of the two-car model.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema. Not validated here: a
        missing column raises ``KeyError`` and an unknown tyre or condition
        gives NaN, so callers validate first (as `probability` and
        `speed_trace` do).
    car : CarParameters, optional
        Physical parameters shared by both cars (default `CAR`).

    Returns
    -------
    dict
        ``attacker`` and ``defender`` (`overtake.physics.CarState`), corner-exit
        speeds ``v0_a`` and ``v0_d`` (m/s), ``gap`` (s), ``length`` (m) and
        relative tyre grip ``grip_a`` and ``grip_d``; arrays have one entry per
        row of ``df``.
    """
    cond = df["track_condition"].to_numpy()
    temp = df["track_temp_c"].to_numpy(dtype=float)
    gap = df["gap_s"].to_numpy(dtype=float)
    grip_a = tyre_grip(df["attacker_tyre"], df["attacker_tyre_age"], cond, temp)
    grip_d = tyre_grip(df["defender_tyre"], df["defender_tyre_age"], cond, temp)

    om = df["overtake_mode"].to_numpy(dtype=bool)
    boost_a = df["attacker_boost"].to_numpy(dtype=bool)
    boost_d = df["defender_boost"].to_numpy(dtype=bool)
    # km/h -> m/s. Car differences are applied to the attacker only; the defender is the reference car.
    speed_gain = df["top_speed_delta_kph"].to_numpy(dtype=float) / 3.6
    # At fixed power P = 1/2 rho CdA v^3, so top speed scales with CdA^(-1/3); a car dv faster at
    # TOP_SPEED_REF_MPS therefore has its CdA scaled by (v_ref / (v_ref + dv))^3.
    cda_a = car.cda_m2 * (TOP_SPEED_REF_MPS / (TOP_SPEED_REF_MPS + speed_gain)) ** 3

    # Boost deploys the full MGU-K limit, otherwise the standard map. Battery % -> J of usable
    # energy; Overtake Mode adds OVERTAKE_MODE_EXTRA_J (0.5 MJ) on top for the attacker (2026 rules).
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
    # Corner speed scales with sqrt(grip) (cornering limit v^2 / r = mu g). Following closely
    # costs grip (dirty air): DIRTY_AIR_GRIP_LOSS right behind, decaying with an e-folding gap of
    # 1 s (estimate). A lap-time advantage is applied as a proportional speed advantage.
    v0_d = EXIT_SPEED_MPS * np.sqrt(grip_d)
    v0_a = (EXIT_SPEED_MPS * np.sqrt(grip_a * (1 - DIRTY_AIR_GRIP_LOSS * np.exp(-gap)))
            * (1 + df["pace_delta_s"].to_numpy(dtype=float) / LAP_TIME_REF_S))
    return {"attacker": attacker, "defender": defender, "v0_a": v0_a, "v0_d": v0_d, "gap": gap,
            "length": df["run_length_m"].to_numpy(dtype=float), "grip_a": grip_a, "grip_d": grip_d}


def closing_run(df, dt=TIME_STEP_S, method="rk4", car=CAR, crossing="hermite"):
    """Stage 1: integrate both cars along the run for every scenario.

    All rows are integrated together as one vectorised system.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema. Not validated here (see
        `initial_conditions`); `probability` validates first.
    dt : float, optional
        Integration time step, s (default `TIME_STEP_S`).
    method : {"rk4", "euler"}, optional
        Integrator; Euler is kept as the first-order baseline.
    car : CarParameters, optional
        Physical parameters shared by both cars (default `CAR`).
    crossing : {"hermite", "linear"}, optional
        Interpolation used to locate the braking point within the last step
        (see `overtake.physics.run_to_braking_point`).

    Returns
    -------
    run : overtake.physics.RunResult
        State at the braking point, including the deficit in seconds.
    grip_a, grip_d : numpy.ndarray
        Relative tyre grip of the attacker and defender, one per row of ``df``.

    Raises
    ------
    RuntimeError
        If a scenario has not reached the braking point within 60 s of
        simulated time (from `overtake.physics.run_to_braking_point`), for
        example because an unknown defender tyre or track condition made the
        defender's speed NaN.
    ValueError
        If ``crossing`` is not "hermite" or "linear".
    KeyError
        If ``method`` is not "rk4" or "euler", or a column of ``df`` is missing.
    """
    ic = initial_conditions(df, car)
    run = run_to_braking_point(ic["length"], ic["gap"], ic["v0_a"], ic["v0_d"], ic["attacker"], ic["defender"],
                               car, dt=dt, method=method, crossing=crossing)
    return run, ic["grip_a"], ic["grip_d"]


def speed_trace(scenario, dt=TIME_STEP_S, car=CAR):
    """Time history of both cars along the run for one scenario.

    Integrates the same equations as `closing_run` with RK4, one step at a time,
    and records every step for plots (`overtake.experiments.speed_trace`).

    Parameters
    ----------
    scenario : dict
        A complete scenario in the `overtake.features` schema.
    dt : float, optional
        RK4 time step, s (default `TIME_STEP_S`).
    car : CarParameters, optional
        Physical parameters shared by both cars (default `CAR`).

    Returns
    -------
    pandas.DataFrame
        Columns, in order: ``t`` (s), ``attacker_x``, ``defender_x`` (m, from
        the corner exit), ``attacker_kph``, ``attacker_mj``, ``defender_kph``,
        ``defender_mj`` (speed in km/h and electrical energy used in MJ). One
        row per step from t = 0 to the last step before the defender reaches
        the braking point. The crossing is not interpolated (unlike
        `closing_run`), so the last row can fall up to one step short (about
        4.6 m at 330 km/h with the default step). ``attacker_x`` starts negative
        because the attacker begins ``gap_s`` behind the defender, who starts
        at the corner exit (x = 0).

    Raises
    ------
    ValueError
        If the scenario does not match the schema (`overtake.features.validate`).

    Examples
    --------
    >>> from overtake.features import DEFAULTS
    >>> trace = speed_trace(DEFAULTS)
    >>> list(trace.columns)
    ['t', 'attacker_x', 'defender_x', 'attacker_kph', 'attacker_mj', 'defender_kph', 'defender_mj']
    >>> len(trace)
    214
    """
    df = pd.DataFrame([scenario])
    validate(df)
    ic = initial_conditions(df, car)
    # One-column state in the layout of overtake.physics.two_car_rhs: attacker x, v, e, then defender
    # x, v, e. The defender starts at the corner exit, the attacker gap * v0_d metres behind it.
    y = np.array([[-ic["gap"][0] * ic["v0_d"][0]], [ic["v0_a"][0]], [0.0], [0.0], [ic["v0_d"][0]], [0.0]])
    rows, t = [], 0.0
    # No time limit, unlike run_to_braking_point: validated inputs give the defender a positive
    # exit speed, and engine power keeps it moving towards the braking point.
    while y[3, 0] < ic["length"][0]:
        rows.append([t, *y[:, 0]])
        y = rk4_step(lambda s: two_car_rhs(s, ic["attacker"], ic["defender"], car), y, dt)
        t += dt
    trace = pd.DataFrame(rows, columns=["t", "attacker_x", "attacker_v", "attacker_e",
                                        "defender_x", "defender_v", "defender_e"])
    # Replace the raw columns with m/s -> km/h and J -> MJ, for plotting.
    for who in ("attacker", "defender"):
        trace[f"{who}_kph"] = trace.pop(f"{who}_v") * 3.6
        trace[f"{who}_mj"] = trace.pop(f"{who}_e") / 1e6
    return trace


def outcome_parameters(df, run, grip_a, grip_d):
    """Stage 2 inputs: effective deficit, execution noise and completion logit.

    Each output has one entry per row of ``df``; together they are the
    arguments of `overtake_integral` and `draw_outcomes`.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema (the same rows as ``run``).
        Not validated here: an unknown category gives NaN.
    run : overtake.physics.RunResult
        Output of `closing_run`.
    grip_a, grip_d : numpy.ndarray
        Relative tyre grip of the attacker and defender (from `closing_run`).

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
    # Weather caution, plus defensive positioning: a defender rated 60 adds nothing, one rated
    # 100 adds 0.04 s (estimate).
    deficit = run.deficit_s + _lookup(CAUTION_S, cond) + 0.04 * (skill_d - 60) / 40
    # Lower-rated attackers are less consistent: a rating of 90 or more adds nothing to the spread,
    # 60 or less adds 0.04 s (estimate).
    sigma = _lookup(EXECUTION_SD, cond) + 0.04 * np.clip((90 - skill_a) / 30, 0, 1)
    z0 = (
        # Equal drivers and grip, a straight, a dry track, the generic circuit and no defender
        # Boost energy: logistic(0.5) = 62 % at zero margin (estimate).
        0.5
        # A 40-point rating difference (the 60-100 training span) shifts the log-odds by 1.2 (estimate).
        + 1.2 * (skill_a - skill_d) / 40
        # Braking grip: the default 5-lap soft (0.991) against the 15-lap medium (0.9715) at 35 C
        # adds 0.49 (estimate).
        + 25.0 * (grip_a - grip_d)
        + _lookup(SEGMENT_LOGIT, df["segment"])
        + _lookup(CONDITION_LOGIT, cond)
        + _lookup(PASSABILITY, df["circuit"])
        # A defender with energy left can fight back on the exit: up to -0.2 with Boost and a
        # full battery (estimate).
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
    deficit, sigma, z0 : array_like
        From `outcome_parameters`; any shapes that broadcast together.
        ``deficit`` and ``sigma`` in s, ``z0`` in log-odds.
    rule : {"gauss", "simpson", "trapezoid"}, optional
        Quadrature rule for ``[0, m]``: Gauss-Legendre with ``n`` nodes, or
        composite Simpson / trapezoid with ``n`` intervals (``n`` even for
        Simpson). Simpson and trapezoid are kept for comparison.
    n : int, optional
        Number of Gauss-Legendre nodes, or of Simpson / trapezoid intervals
        (default 16). With Gauss-Legendre, 16 nodes agree with a 200-node
        reference to within 2e-15 (`overtake.experiments.quadrature`).

    Returns
    -------
    numpy.ndarray
        Probability in [0, 1], with the broadcast shape of ``deficit``,
        ``sigma`` and ``z0`` (a NumPy scalar for scalar inputs).

    Raises
    ------
    ValueError
        If ``rule`` is unknown, or ``rule="simpson"`` with an odd ``n``.
    """
    # New trailing axis: each scenario is evaluated at every quadrature node along the last axis.
    deficit, sigma, z0 = (np.asarray(a, dtype=float)[..., None] for a in (deficit, sigma, z0))
    shift = deficit - LUNGE_RANGE_S  # eps = u + shift, from u = eps - (D - L)

    if rule == "gauss":
        # On [0, m], min(u, m) = u, so the integrand is smooth (the kink of min sits at the end
        # point) and Gauss-Legendre converges fast. Map the nodes from [-1, 1] to [0, m]; the
        # Jacobian m/2 scales the weights.
        nodes, weights = np.polynomial.legendre.leggauss(n)
        u = 0.5 * MAX_MARGIN_S * (nodes + 1)
        w = 0.5 * MAX_MARGIN_S * weights
    else:
        u = np.linspace(0, MAX_MARGIN_S, n + 1)
        h = MAX_MARGIN_S / n
        if rule == "trapezoid":
            # Trapezoid weights: h * [1/2, 1, ..., 1, 1/2].
            w = np.full(n + 1, h)
            w[[0, -1]] = h / 2
        elif rule == "simpson":
            if n % 2:
                raise ValueError("Simpson's rule needs an even number of intervals")
            # Composite Simpson weights: h/3 * [1, 4, 2, 4, ..., 2, 4, 1].
            w = np.where(np.arange(n + 1) % 2, 4.0, 2.0) * h / 3
            w[[0, -1]] = h / 3
        else:
            raise ValueError(f"unknown rule {rule!r}")

    # Normal density phi_sigma of the execution noise at eps = u + shift.
    density = np.exp(-0.5 * ((u + shift) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    body = (density * expit(z0 + MARGIN_LOGIT_PER_S * u) * w).sum(axis=-1)
    # Beyond u = m the logistic is constant, so the tail is that constant times
    # P(eps > m + shift) = 1 - Phi((m + shift) / sigma).
    tail = expit(z0 + MARGIN_LOGIT_PER_S * MAX_MARGIN_S) * (1 - ndtr((MAX_MARGIN_S + shift) / sigma))
    return body + tail[..., 0]  # [..., 0] drops the size-1 node axis of the tail


def probability(df, dt=TIME_STEP_S, car=CAR):
    """Exact probability of an overtake for each scenario (physics + quadrature).

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema.
    dt : float, optional
        ODE time step, s (default `TIME_STEP_S`).
    car : CarParameters, optional
        Physical parameters of both cars (default `CAR`; varied by
        `overtake.experiments.constants`).

    Returns
    -------
    numpy.ndarray
        P(overtake) per scenario, shape ``(len(df),)``, in [0, 1]. Exact up to
        the ODE step error (at most 0.06 ms of deficit at the default step in
        `overtake.experiments.timestep`) and the quadrature error (about
        2e-15, `overtake.experiments.quadrature`).

    Raises
    ------
    ValueError
        If ``df`` does not match the schema (`overtake.features.validate`):
        without this check an unknown category would give NaN silently.
    RuntimeError
        If a scenario's defender has not reached the braking point within 60 s
        of simulated time (from `overtake.physics.run_to_braking_point`).

    Examples
    --------
    >>> import pandas as pd
    >>> from overtake.features import DEFAULTS
    >>> probability(pd.DataFrame([DEFAULTS])).round(3)
    array([0.474])
    """
    validate(df)
    run, grip_a, grip_d = closing_run(df, dt=dt, car=car)
    return overtake_integral(*outcome_parameters(df, run, grip_a, grip_d))


def draw_outcomes(deficit, sigma, z0, rng):
    """Play out one attempt per scenario by drawing the random variables.

    Parameters
    ----------
    deficit, sigma, z0 : numpy.ndarray
        From `outcome_parameters`, all of the same shape.
    rng : numpy.random.Generator
        Source of the execution noise and of the stick test: one normal and
        one uniform draw per scenario.

    Returns
    -------
    numpy.ndarray of bool
        True where the overtake stuck, same shape as the inputs.
    """
    eps = rng.normal(0.0, sigma)
    # Margin u from the module docstring; a lunge is possible when u >= 0.
    margin = eps - (deficit - LUNGE_RANGE_S)
    # Stick test drawn for every scenario (vectorised); the number of draws then does not depend
    # on which scenarios can lunge, so the random stream is fixed.
    sticks = rng.random(margin.shape) < expit(z0 + MARGIN_LOGIT_PER_S * np.minimum(margin, MAX_MARGIN_S))
    return (margin >= 0) & sticks


def monte_carlo_probability(df, n_runs, seed=1):
    """Estimate each scenario's probability by replaying it ``n_runs`` times.

    Used to check `probability`; its standard error is ``sqrt(p (1-p) / n_runs)``.
    The closing run is integrated once per scenario (default `TIME_STEP_S`
    and `CAR`); only the random execution stage is replayed.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios in the `overtake.features` schema. Not validated here: an
        unknown circuit, segment or attacker tyre gives NaN parameters, and
        those replays count as failures.
    n_runs : int
        Replays per scenario. Memory grows as ``len(df) * n_runs``.
    seed : int, optional
        Seed for the replays.

    Returns
    -------
    numpy.ndarray
        Share of replays in which the overtake stuck, one value per scenario.

    Raises
    ------
    RuntimeError
        From `closing_run`, for example when an unknown defender tyre or track
        condition leaves the defender's speed NaN.
    """
    rng = np.random.default_rng(seed)
    params = outcome_parameters(df, *closing_run(df))
    # Copy each scenario's parameters into n_runs columns, so one vectorised call replays them all.
    repeated = [np.repeat(a[:, None], n_runs, axis=1) for a in params]
    return draw_outcomes(*repeated, rng).mean(axis=1)


# Spread of the gap at the Overtake Mode detection point around the gap when
# the attack starts (s): the two are measured at different places on the lap (estimate).
DETECTION_GAP_SD_S = 0.2


def sample_scenarios(n, rng):
    """Draw ``n`` varied race situations spanning the training ranges.

    The ranges are narrower than the hard limits in
    `overtake.features.NUMERIC_RANGES`: gap 0.05-2 s, tyre age 0-40 laps,
    ratings 60-100, top-speed advantage within +/-15 km/h, pace advantage
    within +/-1.5 s/lap, and the run length and track temperature ranges in
    `overtake.features.TRAINED_RUN_LENGTH_M` and
    `overtake.features.TRAINED_TRACK_TEMP_C`. Batteries cover 0-100 %.
    `overtake.predict.OvertakePredictor.out_of_range` flags inputs outside
    these ranges (with 2 % slack).

    Parameters
    ----------
    n : int
        Number of scenarios.
    rng : numpy.random.Generator
        Source of randomness. The order of the draws below fixes the dataset
        for a given seed; reordering them changes every downstream result.

    Returns
    -------
    pandas.DataFrame
        ``n`` rows in the `overtake.features` schema (no label).
    """
    # 75 % dry, 13 % damp, 12 % wet attempts, in the order of CONDITIONS (data design, estimate).
    cond = rng.choice(CONDITIONS, n, p=[0.75, 0.13, 0.12])
    # Compound mix per condition, in the order of COMPOUNDS: soft, medium, hard, intermediate, wet (estimate).
    tyre_probs = {
        "dry": [0.30, 0.38, 0.26, 0.04, 0.02],
        "damp": [0.10, 0.10, 0.08, 0.57, 0.15],
        "wet": [0.03, 0.03, 0.04, 0.48, 0.42],
    }

    def tyres():
        """Draw one compound per scenario from the mix for its track condition."""
        out = np.empty(n, dtype=object)
        for c in CONDITIONS:
            mask = cond == c
            out[mask] = rng.choice(COMPOUNDS, mask.sum(), p=tyre_probs[c])
        return out

    def battery():
        """Draw a charge (%) per scenario: 100 with probability FULL_BATTERY_SHARE, else uniform on 0-100."""
        # A full battery is a common race state (after a safety car or a
        # harvesting lap), so 15 % of attempts start at exactly 100 %. Without
        # this, 100 % sits at the edge of the training data, where the network
        # was found to be least accurate (experiments.sensitivity).
        return np.where(rng.random(n) < FULL_BATTERY_SHARE, 100.0, rng.uniform(0, 100, n))

    # Where the move is completed, in the order of SEGMENTS: most down a straight (data design, estimate).
    seg = rng.choice(SEGMENTS, n, p=[0.55, 0.20, 0.15, 0.10])
    gap = rng.uniform(0.05, 2.0, n)  # attack gap, s (training range)
    # Overtake Mode follows the detection-point gap, so near 1 s it is sometimes on just beyond,
    # and off just within, a 1 s attack gap.
    detection_gap = gap + rng.normal(0.0, DETECTION_GAP_SD_S, n)
    # np.where below draws both options for every row and keeps one, so the random stream does not
    # depend on the segment or weather mix.
    return pd.DataFrame({
        "track_condition": cond,
        "segment": seg,
        "circuit": rng.choice(CIRCUITS, n),  # uniform over the 25 circuits
        "attacker_tyre": tyres(),
        "defender_tyre": tyres(),
        "gap_s": gap,
        "run_length_m": np.where(seg == "straight", rng.uniform(*TRAINED_RUN_LENGTH_M["straight"], n),
                                 rng.uniform(*TRAINED_RUN_LENGTH_M["corner"], n)),
        "attacker_battery": battery(),
        "defender_battery": battery(),
        "attacker_tyre_age": rng.integers(0, 41, n).astype(float),  # 0-40 laps (upper bound exclusive)
        "defender_tyre_age": rng.integers(0, 41, n).astype(float),
        "track_temp_c": np.where(cond == "dry", rng.uniform(*TRAINED_TRACK_TEMP_C["dry"], n),
                                 rng.uniform(*TRAINED_TRACK_TEMP_C["wet"], n)),
        "attacker_skill": rng.uniform(60, 100, n),
        "defender_skill": rng.uniform(60, 100, n),
        "top_speed_delta_kph": np.clip(rng.normal(0, 5, n), -15, 15),  # ~N(0, 5 km/h), clipped (estimate)
        "pace_delta_s": np.clip(rng.normal(0, 0.4, n), -1.5, 1.5),  # ~N(0, 0.4 s/lap), clipped (estimate)
        # Attackers use Boost in 80 % of attempts, defenders in 50 % (estimate).
        "attacker_boost": rng.random(n) < 0.8,
        "defender_boost": rng.random(n) < 0.5,
        "overtake_mode": detection_gap <= 1.0,  # 2026 rule: armed within 1 s at the detection point
    })


def generate_dataset(n, seed=0, chunk=20_000):
    """Sample ``n`` scenarios and simulate one attempt each.

    Parameters
    ----------
    n : int
        Number of attempts.
    seed : int, optional
        Seed of the single generator used for both the scenarios and the
        outcomes.
    chunk : int, optional
        Scenarios integrated together; bounds memory and keeps arrays
        cache-friendly (see `overtake.experiments.performance`). ``p_true``
        does not depend on it, but the 0/1 labels do: outcomes are drawn
        chunk by chunk from one generator, so another chunk size uses the
        random stream in a different order. Keep the default 20,000 to
        reproduce the reported datasets.

    Returns
    -------
    pandas.DataFrame
        ``n`` rows: the 19 scenario columns plus ``overtake`` (0/1 outcome)
        and ``p_true`` (exact probability, for validation only and never a
        model input).

    Examples
    --------
    >>> df = generate_dataset(1000, seed=0)
    >>> df.shape
    (1000, 21)
    """
    rng = np.random.default_rng(seed)
    df = sample_scenarios(n, rng)
    labels, p_true = np.empty(n, dtype=int), np.empty(n)
    # Integrate each chunk once; its stage-2 parameters give both the sampled label and the exact p_true.
    for start in range(0, n, chunk):
        part = df.iloc[start:start + chunk]
        params = outcome_parameters(part, *closing_run(part))
        labels[start:start + chunk] = draw_outcomes(*params, rng)
        p_true[start:start + chunk] = overtake_integral(*params)
    df[LABEL] = labels
    df["p_true"] = p_true
    return df
