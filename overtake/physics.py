"""Straight-line dynamics of two 2026 F1 cars running nose to tail.

The closing stage of an overtaking attempt is an initial value problem. Each
car has position ``x`` (m), speed ``v`` (m/s) and electrical energy used ``e``
(J), and obeys

    dx/dt = v
    m dv/dt = min(F_power, F_traction) - F_drag - F_rolling
    de/dt = P_ers

where

* ``F_power = eta (P_ice + P_ers) / v`` is the drive force the power unit can
  supply. ``P_ers`` is the MGU-K power, limited by the 2026 speed taper, by
  Boost / Overtake Mode, and by the energy left in the battery.
* ``F_traction = mu * w_r * (m g + 1/2 rho ClA v^2)`` is the most the driven
  rear tyres can transmit. For a new soft tyre (``mu = 1.7``) this, not power,
  limits acceleration below about 180 km/h on the standard energy map (about
  200 km/h with Boost, and higher on tyres with less grip), which is how tyre
  grip enters the model.
* ``F_drag = 1/2 rho CdA v^2``. The following car's ``CdA`` is reduced by
  the slipstream, which decays with the distance between the cars.
* ``F_rolling = c_rr (m g + 1/2 rho ClA v^2)``.

The two cars are coupled through the slipstream, so both are integrated
together. Integration starts as both cars exit the corner before the run and
stops when the defender reaches the braking point at the end of the run. The
result is how far (in seconds) the attacker is still behind at that moment.

Two explicit fixed-step methods are provided: forward Euler (first order,
used as a baseline) and classical Runge-Kutta (RK4, fourth order, used by the
simulator). Both are verified against analytical solutions in
``tests/test_physics.py`` and `overtake.experiments.solver`. On the full model
the kinks (traction limit, MGU-K taper, energy running out) lower RK4's
observed order to about 2 (`overtake.experiments.timestep`).

Parameter values are approximate public figures for 2026 cars or engineering
estimates; they are collected in `CarParameters` so they can be changed.
"""
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81  # m/s^2
BATTERY_USABLE_J = 4.0e6        # usable battery energy at 100 % charge, J (estimate)
OVERTAKE_MODE_EXTRA_J = 0.5e6   # extra electrical energy with Overtake Mode, J (2026 rules)


@dataclass(frozen=True)
class CarParameters:
    """Physical parameters shared by both cars.

    The class is frozen, so the shared ``CarParameters()`` instance used as a
    default argument throughout this module cannot be changed by accident.
    Vary a parameter with ``dataclasses.replace(CarParameters(), mass_kg=...)``.

    Attributes
    ----------
    mass_kg : float
        Car + driver + fuel, kg (estimate). The 2026 minimum is about 768 kg
        without fuel.
    cda_m2 : float
        Drag area with the active aero in straight mode, m^2 (estimate).
    cla_m2 : float
        Downforce area with the active aero in straight mode, m^2 (estimate).
    air_density : float
        kg/m^3. 1.2 is dry air at sea level and about 20 deg C.
    rolling_coeff : float
        Rolling-resistance coefficient of the tyres (estimate).
    ice_power_w : float
        Internal combustion engine power, W (about 400 kW in 2026).
    mguk_power_w : float
        Maximum MGU-K (electrical) power, W (350 kW in the 2026 rules).
    drivetrain_efficiency : float
        Share of power-unit output that reaches the wheels (estimate).
    mu_reference : float
        Longitudinal friction coefficient of a new soft tyre on a dry track
        (estimate). Other tyres scale this by their relative grip
        (`overtake.simulator.tyre_grip`).
    driven_axle_share : float
        Share of the vertical load carried by the driven (rear) axle (estimate).
    tow_max : float
        Fractional drag reduction for a follower level with or just behind
        the car ahead, the largest the slipstream gives (estimate).
    tow_length_m : float
        Separation over which the slipstream benefit decays by a factor e, m
        (estimate).
    car_length_m : float
        Distance, m, over which the slipstream fades out once the follower
        moves ahead of level, so drag changes continuously (a sudden change
        would reduce the integrator to first-order accuracy).
    energy_smoothing_j : float
        MGU-K power is ramped to zero over the last ``energy_smoothing_j`` of
        available energy, J, rather than cut instantly, which keeps the
        right-hand side continuous for the integrator. The ramp is also the
        fastest mode of the system: the energy left decays at up to
        ``deploy_share * mguk_power_w / energy_smoothing_j`` per second (17.5
        1/s with Boost below 290 km/h), which limits stable steps to about
        0.16 s for RK4 and 0.11 s for Euler (`overtake.experiments.stability`).
    """

    mass_kg: float = 800.0
    cda_m2: float = 1.0
    cla_m2: float = 3.0
    air_density: float = 1.2
    rolling_coeff: float = 0.012
    ice_power_w: float = 400e3
    mguk_power_w: float = 350e3
    drivetrain_efficiency: float = 0.95
    mu_reference: float = 1.7
    driven_axle_share: float = 0.55
    tow_max: float = 0.25
    tow_length_m: float = 20.0
    car_length_m: float = 5.5
    energy_smoothing_j: float = 20e3


# 2026 MGU-K speed taper (km/h): full power up to the first value, falling
# linearly to zero at the second. Overtake Mode moves the start of the taper.
TAPER_STANDARD_KPH = (290.0, 355.0)
TAPER_OVERTAKE_MODE_KPH = (337.0, 355.0)


def mguk_power_limit(speed, overtake_mode, car=CarParameters()):
    """Maximum MGU-K power allowed by the 2026 regulations at a given speed.

    Full power up to 290 km/h (337 km/h with Overtake Mode), falling linearly
    to zero at 355 km/h (`TAPER_STANDARD_KPH`, `TAPER_OVERTAKE_MODE_KPH`).

    Parameters
    ----------
    speed : array_like
        Car speed in m/s.
    overtake_mode : array_like of bool
        Whether Overtake Mode is active, which delays the taper. Broadcast
        against ``speed``.
    car : CarParameters, optional
        Supplies the maximum, ``car.mguk_power_w`` (350 kW by default).

    Returns
    -------
    numpy.ndarray
        Power limit in W, between 0 and ``car.mguk_power_w``, with the
        broadcast shape of ``speed`` and ``overtake_mode`` (a NumPy scalar
        when both are scalars).

    Examples
    --------
    >>> mguk_power_limit(np.array([250.0, 322.5, 355.0]) / 3.6, False)
    array([350000., 175000.,      0.])
    """
    kph = np.asarray(speed) * 3.6  # the rules state the taper in km/h
    start = np.where(overtake_mode, TAPER_OVERTAKE_MODE_KPH[0], TAPER_STANDARD_KPH[0])
    end = TAPER_STANDARD_KPH[1]  # both modes reach zero at the same speed
    return car.mguk_power_w * np.clip((end - kph) / (end - start), 0.0, 1.0)


@dataclass
class CarState:
    """Per-scenario settings of one car for a run along a straight.

    Every attribute is an array with one entry per scenario, shape ``(n,)``.

    Attributes
    ----------
    mu : numpy.ndarray
        Tyre friction coefficient (`CarParameters.mu_reference` times the
        tyre's relative grip in `overtake.simulator`).
    cda_m2 : numpy.ndarray
        Drag area before any slipstream, m^2.
    deploy_share : numpy.ndarray
        Share of the MGU-K limit the driver uses: 1 with Boost, less on the
        standard energy map (0.55 in `overtake.simulator`).
    energy_j : numpy.ndarray
        Electrical energy available for this run, J.
    overtake_mode : numpy.ndarray of bool, optional
        Overtake Mode active (attacker only). Defaults to all False.
    """

    mu: np.ndarray
    cda_m2: np.ndarray
    deploy_share: np.ndarray
    energy_j: np.ndarray
    overtake_mode: np.ndarray = field(default=None)

    def __post_init__(self):
        """Fill a missing ``overtake_mode`` with False, one entry per scenario of ``mu``."""
        if self.overtake_mode is None:
            self.overtake_mode = np.zeros_like(self.mu, dtype=bool)

    def take(self, mask):
        """Return the subset of scenarios selected by a boolean mask or index array.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask of length ``n`` or integer indices into the scenarios.

        Returns
        -------
        CarState
            A new state holding copies of the selected entries.
        """
        return CarState(self.mu[mask], self.cda_m2[mask], self.deploy_share[mask],
                        self.energy_j[mask], self.overtake_mode[mask])


def acceleration(speed, energy_used, state, cda_m2, car=CarParameters()):
    """Longitudinal acceleration and MGU-K power draw of one car.

    Evaluates ``m dv/dt = min(F_power, F_traction) - F_drag - F_rolling`` (see
    the module docstring) for every scenario at once.

    Parameters
    ----------
    speed : numpy.ndarray
        m/s. Values below 1 m/s are treated as 1 m/s to avoid dividing by zero.
    energy_used : numpy.ndarray
        Electrical energy used so far on this run, J.
    state : CarState
        The car's settings (tyre grip, deployment, energy available).
    cda_m2 : numpy.ndarray
        Drag area to use, m^2, i.e. ``state.cda_m2`` already reduced by any slipstream.
    car : CarParameters, optional
        Constants shared by both cars.

    Returns
    -------
    accel : numpy.ndarray
        m/s^2.
    ers_power : numpy.ndarray
        MGU-K power actually drawn from the battery, W, i.e. ``de/dt``.
    """
    v = np.maximum(speed, 1.0)
    # Share of the MGU-K power still allowed: 1 until the last energy_smoothing_j of the energy
    # available, then falling linearly to 0 as the energy runs out (see CarParameters).
    remaining = np.clip((state.energy_j - energy_used) / car.energy_smoothing_j, 0.0, 1.0)
    ers_available = state.deploy_share * mguk_power_limit(v, state.overtake_mode, car) * remaining

    # Downforce adds to the tyres' vertical load, raising both grip and rolling resistance.
    downforce = 0.5 * car.air_density * car.cla_m2 * v**2
    normal_load = car.mass_kg * GRAVITY + downforce
    f_power = car.drivetrain_efficiency * (car.ice_power_w + ers_available) / v
    f_traction = state.mu * car.driven_axle_share * normal_load
    f_drive = np.minimum(f_power, f_traction)

    # When traction-limited the driver uses less power; the engine supplies
    # first and the MGU-K only the remainder. When power-limited this gives
    # back ers_available, so the upper clip only absorbs round-off.
    power_used = f_drive * v / car.drivetrain_efficiency
    ers_power = np.clip(power_used - car.ice_power_w, 0.0, ers_available)

    f_drag = 0.5 * car.air_density * cda_m2 * v**2
    f_rolling = car.rolling_coeff * normal_load
    return (f_drive - f_drag - f_rolling) / car.mass_kg, ers_power


def slipstream_factor(separation_m, car=CarParameters()):
    """Multiplier (<= 1) on the following car's drag area.

    Parameters
    ----------
    separation_m : numpy.ndarray
        Position of the car ahead minus position of the follower, m. Zero
        when the cars are level, negative once the follower is in front.
    car : CarParameters, optional
        Supplies ``tow_max``, ``tow_length_m`` and ``car_length_m``.

    Returns
    -------
    numpy.ndarray
        ``1 - tow_max * exp(-separation / tow_length)`` while the follower is
        behind or level, fading linearly to 1 as it moves from level to one
        car length in front. Lies between ``1 - tow_max`` (0.75 by default,
        when level) and 1.
    """
    sep = np.asarray(separation_m)
    # Both branches equal 1 at sep = 0, so the factor is continuous. np.where evaluates both for every
    # entry, so the exponential sees sep clipped at 0 and cannot overflow for a follower far ahead.
    behind = np.exp(-np.maximum(sep, 0.0) / car.tow_length_m)
    alongside = np.clip(1.0 + sep / car.car_length_m, 0.0, 1.0)
    return 1.0 - car.tow_max * np.where(sep >= 0, behind, alongside)


def two_car_rhs(y, attacker, defender, car=CarParameters()):
    """Right-hand side of the coupled two-car system.

    Only the attacker gets a slipstream (from the defender ahead); the
    defender's drag is never reduced, even if the attacker draws ahead.

    Parameters
    ----------
    y : numpy.ndarray, shape (6, n)
        Rows are attacker position (m), speed (m/s) and energy used (J), then
        the same for the defender; one column per scenario.
    attacker, defender : CarState
        Settings of each car, one entry per scenario.
    car : CarParameters, optional
        Constants shared by both cars.

    Returns
    -------
    numpy.ndarray, shape (6, n)
        Time derivatives of ``y``: speed, acceleration and MGU-K power of the
        attacker, then of the defender.
    """
    xa, va, ea, xd, vd, ed = y
    acc_a, ers_a = acceleration(va, ea, attacker, attacker.cda_m2 * slipstream_factor(xd - xa, car), car)
    acc_d, ers_d = acceleration(vd, ed, defender, defender.cda_m2, car)
    return np.stack([va, acc_a, ers_a, vd, acc_d, ers_d])


def euler_step(f, y, dt):
    """One forward Euler step of ``dy/dt = f(y)``. First order: global error O(dt).

    Parameters
    ----------
    f : callable
        Maps the state array to its time derivative (same shape).
    y : numpy.ndarray
        State at the start of the step.
    dt : float
        Step size, s.

    Returns
    -------
    numpy.ndarray
        State at the end of the step.
    """
    return y + dt * f(y)


def rk4_step(f, y, dt):
    """One classical Runge-Kutta step of ``dy/dt = f(y)``. Fourth order: global error O(dt^4).

    The order holds for a smooth ``f``; each step costs four evaluations of ``f``.

    Parameters
    ----------
    f : callable
        Maps the state array to its time derivative (same shape).
    y : numpy.ndarray
        State at the start of the step.
    dt : float
        Step size, s.

    Returns
    -------
    numpy.ndarray
        State at the end of the step.
    """
    # Slopes at the start, twice at the midpoint and at the end, combined with Simpson weights 1-2-2-1.
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# Step functions selected by the method argument of integrate and run_to_braking_point.
STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def integrate(f, y0, dt, n_steps, method="rk4"):
    """Integrate an autonomous system ``dy/dt = f(y)`` with fixed steps.

    Parameters
    ----------
    f : callable
        Maps the state array to its time derivative (same shape).
    y0 : array_like
        Initial state. It is copied to a float array, so the input is not modified.
    dt : float
        Step size, s.
    n_steps : int
        Number of steps.
    method : {"rk4", "euler"}, optional
        Step function from `STEPPERS`.

    Returns
    -------
    numpy.ndarray
        State after ``n_steps * dt`` seconds.

    Raises
    ------
    KeyError
        If ``method`` is not "rk4" or "euler".
    """
    step = STEPPERS[method]
    y = np.array(y0, dtype=float)
    for _ in range(n_steps):
        y = step(f, y, dt)
    return y


def _hermite(p0, m0, p1, m1, s, dt):
    """Cubic Hermite interpolant at fraction ``s`` in [0, 1] of a step of length ``dt``.

    ``p0`` and ``p1`` are the values and ``m0`` and ``m1`` the time derivatives
    at the two ends of the step.
    """
    # Hermite basis on [0, 1]; the derivative terms are multiplied by dt because s = t / dt.
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * dt * m0 + h01 * p1 + h11 * dt * m1


def _crossing_fraction(x0, v0, x1, v1, target, dt, iterations=4):
    """Fraction of the step at which the Hermite cubic through (x0, v0), (x1, v1) reaches ``target``.

    Starts from linear interpolation and refines with a fixed number of Newton
    iterations (no tolerance test). On the random steps of
    ``tests/test_physics.py`` two iterations already reach round-off (about
    3e-13 m), so the default of four leaves a margin. Hermite interpolation is
    fourth-order accurate, matching RK4.
    """
    # Linear first guess; the 1e-12 m floor avoids dividing by zero if the car did not move.
    s = np.clip((target - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
    for _ in range(iterations):
        value = _hermite(x0, v0, x1, v1, s, dt) - target
        # d/ds of the interpolant: the derivatives of the four basis functions in _hermite.
        slope = (6 * s**2 - 6 * s) * x0 + (3 * s**2 - 4 * s + 1) * dt * v0 \
            + (-6 * s**2 + 6 * s) * x1 + (3 * s**2 - 2 * s) * dt * v1
        # Newton update, guarded against a zero slope and kept inside the step.
        s = np.clip(s - value / np.where(np.abs(slope) > 1e-12, slope, 1e-12), 0.0, 1.0)
    return s


@dataclass
class RunResult:
    """State of both cars when the defender reaches the braking point.

    Array attributes have one entry per scenario, in input order.

    Attributes
    ----------
    deficit_s : numpy.ndarray
        Gap the attacker still has to close, s (distance behind / attacker
        speed, the speed floored at 1 m/s). Negative means the attacker is
        already ahead.
    attacker_speed, defender_speed : numpy.ndarray
        Speeds at the braking point, m/s.
    attacker_energy_j, defender_energy_j : numpy.ndarray
        Electrical energy used on the run, J.
    time_s : numpy.ndarray
        Time the defender took to cover the run, s.
    steps : int
        Integration steps taken (until the slowest scenario finished).
    """

    deficit_s: np.ndarray
    attacker_speed: np.ndarray
    defender_speed: np.ndarray
    attacker_energy_j: np.ndarray
    defender_energy_j: np.ndarray
    time_s: np.ndarray
    steps: int


def run_to_braking_point(run_length_m, gap_s, attacker_speed0, defender_speed0, attacker, defender,
                         car=CarParameters(), dt=0.05, method="rk4", max_time_s=60.0, crossing="hermite"):
    """Integrate both cars from the corner exit until the defender reaches the braking point.

    All scenarios are integrated together as NumPy arrays. Scenarios drop out
    of the active set as soon as their defender crosses the braking point; the
    exact crossing time within the last step is found by Hermite interpolation.

    Parameters
    ----------
    run_length_m : array_like, shape (n,)
        Distance from the corner exit to the braking point, m. Must be > 0.
    gap_s : array_like, shape (n,)
        Time gap at the corner exit, s. The attacker starts ``gap_s *
        defender_speed0`` metres behind.
    attacker_speed0, defender_speed0 : array_like, shape (n,)
        Corner-exit speeds, m/s.
    attacker, defender : CarState
        Settings of each car, one entry per scenario.
    car : CarParameters, optional
        Constants shared by both cars.
    dt : float, optional
        Time step, s. At 0.05 s the worst deficit error over 2000 random
        scenarios is about 0.06 ms against a 0.001 s reference, well inside
        the 1 ms limit (`overtake.experiments.timestep`), and the step is
        about three times smaller than RK4's stability limit
        (`overtake.experiments.stability`).
    method : {"rk4", "euler"}, optional
        Step function from `STEPPERS`.
    max_time_s : float, optional
        Safety limit, s; a RuntimeError is raised if a defender has not
        reached the braking point by then (e.g. a non-physical input left it
        stationary).
    crossing : {"hermite", "linear"}, optional
        How the state at the braking point is found within the last step:
        cubic Hermite interpolation with a Newton solve (fourth order, the
        default) or straight-line interpolation (second order; kept only so
        that `overtake.experiments.timestep` can measure the difference).

    Returns
    -------
    RunResult
        State of both cars at the braking point, one entry per scenario.

    Raises
    ------
    ValueError
        If ``crossing`` is not "hermite" or "linear".
    KeyError
        If ``method`` is not "rk4" or "euler".
    RuntimeError
        If any scenario is still running after ``max_time_s``.

    Examples
    --------
    An attacker 0.6 s behind, with Boost and Overtake Mode, against a defender
    on the standard energy map, on a 400 m and a 1000 m run:

    >>> n = 2
    >>> attacker = CarState(mu=np.full(n, 1.7), cda_m2=np.ones(n), deploy_share=np.ones(n),
    ...                     energy_j=np.full(n, 4.5e6), overtake_mode=np.ones(n, dtype=bool))
    >>> defender = CarState(mu=np.full(n, 1.7), cda_m2=np.ones(n), deploy_share=np.full(n, 0.55),
    ...                     energy_j=np.full(n, 4.0e6))
    >>> result = run_to_braking_point(np.array([400.0, 1000.0]), np.full(n, 0.6), np.full(n, 33.0),
    ...                               np.full(n, 33.0), attacker, defender)
    >>> result.deficit_s.round(3)
    array([-0.064, -0.636])
    """
    if crossing not in ("hermite", "linear"):
        raise ValueError(f"crossing must be 'hermite' or 'linear', not {crossing!r}")
    length = np.asarray(run_length_m, dtype=float)
    n = length.size
    # Rows as in two_car_rhs. The defender starts at the corner exit (x = 0), the attacker the
    # time gap times the defender's speed behind it, and neither has used any energy yet.
    y = np.stack([
        -np.asarray(gap_s, dtype=float) * defender_speed0,
        np.asarray(attacker_speed0, dtype=float),
        np.zeros(n),
        np.zeros(n),
        np.asarray(defender_speed0, dtype=float),
        np.zeros(n),
    ])
    final = np.full((6, n), np.nan)  # state at the braking point, filled in as each scenario finishes
    finish_time = np.full(n, np.nan)
    # Original indices of the scenarios still running; att, dfd, target and y shrink with it.
    active = np.arange(n)
    att, dfd, target = attacker, defender, length
    step = STEPPERS[method]
    t, steps = 0.0, 0

    def f(state):
        """Right-hand side for the active scenarios, using ``att`` and ``dfd`` as last rebound in the loop."""
        return two_car_rhs(state, att, dfd, car)

    while active.size:
        if t > max_time_s:
            raise RuntimeError(f"{active.size} scenario(s) did not reach the braking point in {max_time_s} s")
        y_new = step(f, y, dt)
        crossed = y_new[3] >= target  # the run ends when the defender reaches the braking point
        if crossed.any():
            idx = active[crossed]  # columns of the finishing scenarios in the full-size outputs
            if crossing == "linear":
                # Straight line between the step's end points: second-order accurate.
                s = np.clip((target[crossed] - y[3, crossed]) / np.maximum(y_new[3, crossed] - y[3, crossed], 1e-12),
                            0.0, 1.0)
                final[:, idx] = y[:, crossed] + s * (y_new[:, crossed] - y[:, crossed])
            else:
                # The derivatives at both ends of the step make the interpolant cubic and
                # fourth-order accurate, matching RK4. The fraction s is solved on the
                # defender's position, then used to interpolate all six states.
                att_x, dfd_x = att.take(crossed), dfd.take(crossed)
                d_old = two_car_rhs(y[:, crossed], att_x, dfd_x, car)
                d_new = two_car_rhs(y_new[:, crossed], att_x, dfd_x, car)
                s = _crossing_fraction(y[3, crossed], y[4, crossed], y_new[3, crossed], y_new[4, crossed],
                                       target[crossed], dt)
                for row in range(6):
                    final[row, idx] = _hermite(y[row, crossed], d_old[row], y_new[row, crossed], d_new[row], s, dt)
            finish_time[idx] = t + s * dt
            # Drop finished scenarios so later steps integrate only those still running.
            keep = ~crossed
            active, y_new, target = active[keep], y_new[:, keep], target[keep]
            att, dfd = att.take(keep), dfd.take(keep)
        y = y_new
        t += dt
        steps += 1

    xa, va, ea, xd, vd, ed = final
    return RunResult(
        deficit_s=(xd - xa) / np.maximum(va, 1.0),
        attacker_speed=va,
        defender_speed=vd,
        attacker_energy_j=ea,
        defender_energy_j=ed,
        time_s=finish_time,
        steps=steps,
    )
