"""Checks on helpers used by the experiments."""
import pandas as pd
import pytest

from overtake.experiments import nn_flops, tyre_mismatch
from overtake.features import DEFAULTS


def test_flop_count_matches_layer_sizes():
    # 33 inputs -> 64 -> 64 -> 1: multiply-adds (2 FLOPs), bias adds and ~5 FLOPs per SiLU.
    expected = (2 * 33 * 64 + 64) + (2 * 64 * 64 + 64) + (2 * 64 * 1 + 1) + 5 * (64 + 64)
    assert nn_flops((64, 64), 33) == expected


def test_tyre_mismatch_rule():
    df = pd.DataFrame([
        DEFAULTS,                                                                      # slicks, dry
        {**DEFAULTS, "attacker_tyre": "intermediate"},                                 # treaded, dry
        {**DEFAULTS, "track_condition": "wet", "attacker_tyre": "wet", "defender_tyre": "intermediate"},
        {**DEFAULTS, "track_condition": "damp", "attacker_tyre": "intermediate"},      # defender on slicks
    ])
    assert tyre_mismatch(df).tolist() == [False, True, False, True]


def test_rk4_stability_limit_on_the_real_axis():
    """|R(z)| = 1 at z = -2.785 for RK4, so the battery ramp limits dt to about 0.16 s with Boost."""
    from scipy.optimize import brentq
    from overtake.experiments import rk4_amplification
    limit = -brentq(lambda z: rk4_amplification(z) - 1, -3.0, -2.0)
    assert limit == pytest.approx(2.7853, abs=1e-4)
    assert abs(rk4_amplification(-2.0)) < 1 < abs(rk4_amplification(-3.0))


def test_patched_restores_constants_even_after_an_error():
    """The constant-sensitivity studies must leave the simulator's constants as they found them."""
    from overtake import simulator
    from overtake.experiments import _patched
    before = simulator.EXIT_SPEED_MPS
    with pytest.raises(RuntimeError):
        with _patched(simulator, EXIT_SPEED_MPS=1.0):
            assert simulator.EXIT_SPEED_MPS == 1.0
            raise RuntimeError
    assert simulator.EXIT_SPEED_MPS == before


def test_overtake_mode_energy_matters_only_when_the_battery_is_short():
    """A full battery already covers an 800 m straight, so the 0.5 MJ bonus changes nothing; at 30 % it helps."""
    from overtake import simulator
    from overtake.experiments import _patched
    scenarios = pd.DataFrame([{**DEFAULTS, "attacker_battery": b} for b in (100.0, 30.0)])
    with _patched(simulator, OVERTAKE_MODE_EXTRA_J=0.0):
        without = simulator.probability(scenarios)
    with_bonus = simulator.probability(scenarios)
    assert with_bonus[0] == pytest.approx(without[0], abs=1e-6)
    assert with_bonus[1] > without[1] + 0.01
