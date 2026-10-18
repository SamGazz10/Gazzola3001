import pandas as pd
import pytest

from overtake.features import DEFAULTS
from overtake.simulator import true_probability


def prob(**changes):
    return true_probability(pd.DataFrame([{**DEFAULTS, **changes}]), n_runs=20_000)[0]


@pytest.mark.parametrize("better, worse", [
    ({}, {"attacker_boost": False}),
    ({"gap_s": 0.5}, {"gap_s": 1.0}),
    ({"attacker_battery": 100}, {"attacker_battery": 40}),
    ({"defender_battery": 20}, {"defender_battery": 80}),
    ({"run_length_m": 1100}, {"run_length_m": 500}),
    ({"circuit": "monza"}, {"circuit": "monaco"}),
    ({"attacker_skill": 95}, {"attacker_skill": 70}),
    ({"overtake_mode": True}, {"overtake_mode": False}),
    ({"track_condition": "dry"}, {"track_condition": "wet"}),
])
def test_factors_move_probability_the_right_way(better, worse):
    assert prob(**better) > prob(**worse)


def test_slicks_are_hopeless_in_the_wet():
    assert prob(track_condition="wet", attacker_tyre="soft", defender_tyre="intermediate") < 0.01


def test_one_second_on_a_straight_is_possible_but_not_likely():
    assert 0.05 < prob() < 0.5
