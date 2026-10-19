"""Checks on helpers used by the experiments."""
import pandas as pd

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
