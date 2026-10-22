"""Checks on the overtaking simulator: the exact probability, its agreement with
Monte Carlo replays, and the direction of each factor's effect."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit

from overtake import simulator as sim
from overtake.features import DEFAULTS


def prob(**changes):
    return sim.probability(pd.DataFrame([{**DEFAULTS, **changes}]))[0]


@pytest.mark.parametrize("better, worse", [
    ({}, {"attacker_boost": False}),
    ({"gap_s": 0.5}, {"gap_s": 1.0}),
    ({"attacker_battery": 100}, {"attacker_battery": 40}),
    ({"defender_battery": 20}, {"defender_battery": 80}),
    ({"run_length_m": 1100}, {"run_length_m": 500}),
    ({"circuit": "monza"}, {"circuit": "monaco"}),
    ({"attacker_skill": 95}, {"attacker_skill": 70}),
    ({"overtake_mode": True}, {"overtake_mode": False}),
    ({}, {"track_condition": "wet", "attacker_tyre": "intermediate", "defender_tyre": "intermediate"}),
    ({"defender_boost": False}, {"defender_boost": True}),
])
def test_factors_move_probability_the_right_way(better, worse):
    assert prob(**better) > prob(**worse)


def test_slicks_are_hopeless_in_the_wet():
    assert prob(track_condition="wet", attacker_tyre="soft", defender_tyre="intermediate") < 0.01


@pytest.mark.parametrize("deficit, sigma, z0", [(0.4, 0.2, 0.5), (0.0, 0.25, -1.0), (-0.3, 0.3, 2.0)])
def test_quadrature_matches_adaptive_reference(deficit, sigma, z0):
    def integrand(u):
        density = np.exp(-0.5 * ((u + deficit - sim.LUNGE_RANGE_S) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return density * expit(z0 + sim.MARGIN_LOGIT_PER_S * min(u, sim.MAX_MARGIN_S))

    reference = (quad(integrand, 0, sim.MAX_MARGIN_S, epsabs=1e-13)[0]
                 + quad(integrand, sim.MAX_MARGIN_S, np.inf, epsabs=1e-13)[0])
    value = sim.overtake_integral(np.array([deficit]), np.array([sigma]), np.array([z0]))[0]
    assert value == pytest.approx(reference, abs=1e-10)


def test_monte_carlo_agrees_with_exact_probability():
    df = sim.sample_scenarios(50, np.random.default_rng(5))
    n_runs = 20_000
    exact = sim.probability(df)
    estimate = sim.monte_carlo_probability(df, n_runs)
    standard_error = np.sqrt(exact * (1 - exact) / n_runs) + 1e-4
    assert np.all(np.abs(estimate - exact) < 4.5 * standard_error)


def test_dataset_labels_match_true_probability_on_average():
    df = sim.generate_dataset(20_000, seed=2)
    assert set(df["overtake"].unique()) <= {0, 1}
    assert df["overtake"].mean() == pytest.approx(df["p_true"].mean(), abs=0.01)
    assert df["p_true"].between(0, 1).all()


def test_probability_rejects_an_unknown_category():
    """Without validation an unknown circuit would map to NaN and give P = NaN silently."""
    with pytest.raises(ValueError, match="circuit"):
        sim.probability(pd.DataFrame([{**DEFAULTS, "circuit": "imola"}]))
