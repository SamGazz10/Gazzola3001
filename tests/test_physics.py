"""Verification of the vehicle-dynamics model and its integrators.

The full model has no closed-form solution, so the implemented right-hand side
is checked in special cases that do: constant power with no resistance, and
aerodynamic drag with no power.
"""
from dataclasses import replace

import numpy as np
import pytest

from overtake.physics import (CarParameters, CarState, acceleration, integrate, mguk_power_limit,
                              run_to_braking_point)

M, P, K = 800.0, 500e3, 0.6   # mass (kg), power (W), drag constant 1/2 rho CdA (kg/m)
V0, T = 30.0, 10.0            # initial speed (m/s), end time (s)


def single_car(car):
    state = CarState(mu=np.array([1e9]), cda_m2=np.array([car.cda_m2]),
                     deploy_share=np.array([0.0]), energy_j=np.array([0.0]))

    def f(y):
        accel, ers = acceleration(y[1:2], y[2:3], state, state.cda_m2, car)
        return np.array([y[1], accel[0], ers[0]])
    return f


# Constant power, no resistance: m v dv/dt = P  =>  v^2 = v0^2 + 2 P t / m.
POWER_ONLY = replace(CarParameters(), mass_kg=M, ice_power_w=P, cda_m2=0.0, cla_m2=0.0,
                     rolling_coeff=0.0, drivetrain_efficiency=1.0)
# Drag only: m dv/dt = -K v^2  =>  v = v0 / (1 + K v0 t / m).
DRAG_ONLY = replace(POWER_ONLY, ice_power_w=0.0, cda_m2=2 * K / 1.2, air_density=1.2)


def exact_power_only(t):
    v = np.sqrt(V0**2 + 2 * P * t / M)
    x = M / (3 * P) * (v**3 - V0**3)
    return x, v


def exact_drag_only(t):
    v = V0 / (1 + K * V0 * t / M)
    x = M / K * np.log(1 + K * V0 * t / M)
    return x, v


@pytest.mark.parametrize("car, exact", [(POWER_ONLY, exact_power_only), (DRAG_ONLY, exact_drag_only)])
def test_rk4_matches_analytical_solution(car, exact):
    y = integrate(single_car(car), [0.0, V0, 0.0], dt=0.05, n_steps=int(T / 0.05))
    x, v = exact(T)
    assert y[0] == pytest.approx(x, rel=1e-8)
    assert y[1] == pytest.approx(v, rel=1e-8)


@pytest.mark.parametrize("method, order", [("euler", 1), ("rk4", 4)])
def test_observed_order_of_convergence(method, order):
    f, (x_exact, _) = single_car(POWER_ONLY), exact_power_only(T)
    errors = [abs(integrate(f, [0.0, V0, 0.0], dt, int(round(T / dt)), method)[0] - x_exact)
              for dt in (0.2, 0.1)]
    observed = np.log2(errors[0] / errors[1])
    assert observed == pytest.approx(order, abs=0.15)


def test_mguk_taper_follows_2026_rules():
    kph = np.array([250, 290, 322.5, 355, 360]) / 3.6
    assert mguk_power_limit(kph, False) == pytest.approx([350e3, 350e3, 175e3, 0, 0])
    assert mguk_power_limit(np.array([337, 346]) / 3.6, True) == pytest.approx([350e3, 175e3])


def run(n=200, seed=0, dt=0.05):
    rng = np.random.default_rng(seed)
    car = CarParameters()

    def state(om):
        return CarState(mu=rng.uniform(1.0, 1.7, n), cda_m2=np.full(n, car.cda_m2),
                        deploy_share=rng.choice([0.55, 1.0], n), energy_j=rng.uniform(0, 4.5e6, n),
                        overtake_mode=np.full(n, om))
    args = (rng.uniform(100, 1300, n), rng.uniform(0.1, 2.0, n), rng.uniform(25, 35, n),
            rng.uniform(25, 35, n), state(True), state(False))
    return args, run_to_braking_point(*args, dt=dt)


def test_energy_used_never_exceeds_energy_available():
    (_, _, _, _, attacker, defender), result = run()
    assert np.all(result.attacker_energy_j <= attacker.energy_j + 1.0)
    assert np.all(result.defender_energy_j <= defender.energy_j + 1.0)


def test_speeds_stay_physical():
    _, result = run()
    top_speed = 360 / 3.6  # no car can exceed the MGU-K cut-off speed by much
    for v in (result.attacker_speed, result.defender_speed):
        assert np.all((v > 20) & (v < top_speed))


def test_braking_point_result_converges_with_timestep():
    _, coarse = run(dt=0.05)
    _, fine = run(dt=0.005)
    assert np.max(np.abs(coarse.deficit_s - fine.deficit_s)) < 1e-3


def test_braking_point_time_matches_analytical_solution():
    """The production path (integration plus Hermite crossing) against the constant-power exact solution."""
    length = np.array([300.0, 800.0])
    state = CarState(mu=np.full(2, 1e9), cda_m2=np.zeros(2), deploy_share=np.zeros(2), energy_j=np.zeros(2))
    result = run_to_braking_point(length, np.full(2, 0.5), np.full(2, V0), np.full(2, V0), state, state,
                                  car=POWER_ONLY)
    # x = m / (3P) (v^3 - v0^3) gives the speed at the braking point, then v^2 = v0^2 + 2 P t / m the time.
    v_end = (3 * P * length / M + V0**3) ** (1 / 3)
    assert result.defender_speed == pytest.approx(v_end, rel=1e-8)
    assert result.time_s == pytest.approx(M * (v_end**2 - V0**2) / (2 * P), rel=1e-8)


def test_hermite_crossing_beats_linear_interpolation():
    """Linear interpolation at the braking point is second order; Hermite matches RK4's fourth order."""
    args, fine = run(dt=0.001)
    errors = {crossing: np.abs(run_to_braking_point(*args, dt=0.05, crossing=crossing).deficit_s
                               - fine.deficit_s).max() for crossing in ("hermite", "linear")}
    assert errors["hermite"] < errors["linear"]


def test_unknown_crossing_method_is_rejected():
    """A misspelt interpolation option raises instead of silently using a default."""
    args, _ = run(n=2)
    with pytest.raises(ValueError, match="crossing"):
        run_to_braking_point(*args, crossing="cubic")


def test_newton_crossing_converges():
    """Four Newton iterations from the linear guess solve the Hermite crossing to far below a millimetre."""
    from overtake.physics import _crossing_fraction, _hermite
    rng = np.random.default_rng(1)
    n, dt = 1000, 0.05
    x0, v0 = rng.uniform(0, 1000, n), rng.uniform(30, 95, n)
    v1 = v0 + rng.uniform(-1, 1, n)       # a car's speed changes by at most about 1 m/s in one step
    x1 = x0 + dt * (v0 + v1) / 2
    target = x0 + rng.uniform(0, 1, n) * (x1 - x0)
    s = _crossing_fraction(x0, v0, x1, v1, target, dt)
    assert np.abs(_hermite(x0, v0, x1, v1, s, dt) - target).max() < 1e-9
