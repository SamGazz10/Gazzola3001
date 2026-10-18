"""Physics-informed Monte Carlo simulator of one overtaking attempt under the 2026 rules.

There is no public dataset of 2026 overtakes that records battery state of
charge and Boost / Overtake Mode usage, so the network is trained on attempts
generated here. Each attempt is played out in two stages:

1. Closing: how much of the gap the attacker removes before the braking point.
   Electrical deployment (Boost, Overtake Mode, battery charge on both cars),
   car straight-line speed, slipstream, corner-exit grip and dirty air all
   contribute, plus random execution noise.
2. Completion: if the attacker gets close enough to lunge, whether the move
   sticks, which depends on the margin, driver ratings, braking grip, the
   corner type, the weather and the circuit.

Every constant is a simplified estimate chosen to give plausible magnitudes,
not a value measured from F1 data. Change them to test other assumptions.
"""
import numpy as np
import pandas as pd

from .features import CIRCUITS, COMPOUNDS, CONDITIONS, LABEL, SEGMENTS

# 2026 power unit (approximate public figures).
BATTERY_USABLE_MJ = 4.0         # usable energy at 100% state of charge
MGUK_POWER_MJ_PER_S = 0.35      # 350 kW electrical deployment
OVERTAKE_MODE_EXTRA_MJ = 0.5    # extra energy when within 1 s at the detection point
STANDARD_MAP_SHARE = 0.55       # share of full deployment used on a straight without Boost
FULL_DEPLOYMENT_DV = 14.0       # m/s end-of-run speed swing between full and zero deployment
OVERTAKE_MODE_DV = 1.0          # m/s extra average speed from the Overtake Mode power profile

LUNGE_RANGE_S = 0.25            # gap at the braking point from which a lunge is possible
MAX_MARGIN_S = 0.5

MEAN_SPEED = {"dry": 80.0, "damp": 72.0, "wet": 66.0}      # m/s along the run
TRACTION = {"dry": 1.0, "damp": 0.7, "wet": 0.5}           # share of extra power that becomes speed
TOW = {"dry": 1.0, "damp": 0.8, "wet": 0.6}                # attackers stay out of the spray when wet
EXECUTION_SD = {"dry": 0.28, "damp": 0.32, "wet": 0.36}    # s, attempt-to-attempt variation
CAUTION_S = {"dry": 0.0, "damp": 0.05, "wet": 0.12}       # time given away backing off in poor grip/visibility
CONDITION_LOGIT = {"dry": 0.0, "damp": -0.5, "wet": -0.9}
DIRTY_AIR_S = {"straight": 0.03, "slow_corner": 0.04, "medium_corner": 0.07, "fast_corner": 0.12}
SEGMENT_LOGIT = {"straight": 0.0, "slow_corner": 0.2, "medium_corner": -0.6, "fast_corner": -1.6}

# Base grip of each compound on each surface (1.0 = new soft on a dry track).
GRIP = {
    "dry": {"soft": 1.000, "medium": 0.988, "hard": 0.976, "intermediate": 0.900, "wet": 0.860},
    "damp": {"soft": 0.900, "medium": 0.890, "hard": 0.875, "intermediate": 0.950, "wet": 0.920},
    "wet": {"soft": 0.700, "medium": 0.690, "hard": 0.680, "intermediate": 0.890, "wet": 0.930},
}
DEG_PER_LAP = {"soft": 0.0018, "medium": 0.0011, "hard": 0.0007, "intermediate": 0.0015, "wet": 0.0012}
CLIFF_LAP = {"soft": 22, "medium": 35, "hard": 50, "intermediate": 25, "wet": 30}
CLIFF_DEG_PER_LAP = 0.004
GRIP_TO_PACE = 85.0             # s/lap per unit of grip (0.01 grip ~ 0.85 s/lap)
EXIT_SHARE_OF_LAP = 0.06        # share of a lap spent in the corners leading onto the run

# How readily a move sticks at each circuit once alongside (logit offset).
# Rough estimates from track width, braking zones and overtaking history.
PASSABILITY = {
    "generic": 0.0, "melbourne": -0.1, "shanghai": 0.4, "suzuka": -0.4, "bahrain": 0.6,
    "jeddah": 0.3, "miami": 0.1, "montreal": 0.4, "monaco": -2.5, "barcelona": -0.3,
    "spielberg": 0.4, "silverstone": 0.1, "spa": 0.6, "hungaroring": -1.0, "zandvoort": -1.2,
    "monza": 0.5, "madrid": 0.0, "baku": 0.5, "singapore": -0.9, "austin": 0.3,
    "mexico_city": 0.3, "interlagos": 0.4, "las_vegas": 0.5, "lusail": 0.0, "yas_marina": 0.1,
}
assert set(PASSABILITY) == set(CIRCUITS)


def _lookup(table, keys):
    return pd.Series(keys).map(table).to_numpy(dtype=float)


def tyre_grip(compound, age, condition, track_temp):
    """Grip of a tyre after `age` laps, relative to a new soft on a dry track."""
    compound, condition = np.asarray(compound), np.asarray(condition)
    base = pd.Series(list(zip(condition, compound))).map(
        {(c, t): GRIP[c][t] for c in CONDITIONS for t in COMPOUNDS}
    ).to_numpy(dtype=float)
    deg = _lookup(DEG_PER_LAP, compound) * np.clip(1 + 0.02 * (track_temp - 35), 0.6, 1.6)
    # Treaded tyres overheat and wear quickly on a dry track.
    deg = np.where((condition == "dry") & np.isin(compound, ["intermediate", "wet"]), deg * 3, deg)
    wear = deg * age + CLIFF_DEG_PER_LAP * np.maximum(0, age - _lookup(CLIFF_LAP, compound))
    # Hard tyres struggle to reach temperature on a cold track.
    cold_hard = (compound == "hard") & (track_temp < 25)
    return base - wear - 0.01 * cold_hard


def simulate(df, rng):
    """Play out one attempt per row. Returns a boolean array: did the overtake stick?"""
    n = len(df)
    cond = df["track_condition"].to_numpy()
    seg = df["segment"].to_numpy()
    run = df["run_length_m"].to_numpy(dtype=float)
    gap = df["gap_s"].to_numpy(dtype=float)
    soc_a = df["attacker_battery"].to_numpy(dtype=float) / 100
    soc_d = df["defender_battery"].to_numpy(dtype=float) / 100
    boost_a = df["attacker_boost"].to_numpy(dtype=bool)
    boost_d = df["defender_boost"].to_numpy(dtype=bool)
    om = df["overtake_mode"].to_numpy(dtype=bool)
    skill_a = df["attacker_skill"].to_numpy(dtype=float)
    skill_d = df["defender_skill"].to_numpy(dtype=float)
    temp = df["track_temp_c"].to_numpy(dtype=float)

    # Stage 1: closing.
    speed = _lookup(MEAN_SPEED, cond)
    traction = _lookup(TRACTION, cond)
    energy_needed = MGUK_POWER_MJ_PER_S * run / speed + 1e-6  # MJ to run 350 kW all the way
    energy_a = BATTERY_USABLE_MJ * soc_a + OVERTAKE_MODE_EXTRA_MJ * om
    energy_d = BATTERY_USABLE_MJ * soc_d
    share_a = np.minimum(np.where(boost_a, 1.0, STANDARD_MAP_SHARE), energy_a / energy_needed)
    share_d = np.minimum(np.where(boost_d, 1.0, STANDARD_MAP_SHARE), energy_d / energy_needed)

    s_per_mps = run / speed**2  # seconds gained over the run per m/s of extra average speed
    gain_energy = s_per_mps * 0.5 * FULL_DEPLOYMENT_DV * traction * (share_a - share_d)
    gain_energy *= rng.normal(1.0, 0.1, n)  # deployment efficiency varies lap to lap
    gain_om = s_per_mps * OVERTAKE_MODE_DV * traction * om * np.minimum(1.0, run / 800)
    gain_car = s_per_mps * 0.6 * df["top_speed_delta_kph"].to_numpy(dtype=float) / 3.6
    gain_tow = 0.00022 * run * np.exp(-gap / 0.7) * _lookup(TOW, cond)

    grip_a = tyre_grip(df["attacker_tyre"], df["attacker_tyre_age"].to_numpy(dtype=float), cond, temp)
    grip_d = tyre_grip(df["defender_tyre"], df["defender_tyre_age"].to_numpy(dtype=float), cond, temp)
    pace = GRIP_TO_PACE * (grip_a - grip_d) + df["pace_delta_s"].to_numpy(dtype=float)
    gain_exit = EXIT_SHARE_OF_LAP * pace

    loss_dirty_air = _lookup(DIRTY_AIR_S, seg) * np.exp(-gap)
    loss_defence = 0.04 * (skill_d - 60) / 40
    loss_caution = _lookup(CAUTION_S, cond)

    closing = (gain_energy + gain_om + gain_car + gain_tow + gain_exit
               - loss_dirty_air - loss_defence - loss_caution)
    sd = _lookup(EXECUTION_SD, cond) + 0.04 * np.clip((90 - skill_a) / 30, 0, 1)
    deficit = gap - closing - rng.normal(0.0, sd)  # gap left at the braking point
    alongside = deficit <= LUNGE_RANGE_S

    # Stage 2: completion.
    margin = np.minimum(LUNGE_RANGE_S - deficit, MAX_MARGIN_S)
    logit = (
        0.5
        + 5.0 * margin
        + 1.2 * (skill_a - skill_d) / 40
        + 25.0 * (grip_a - grip_d)
        + _lookup(SEGMENT_LOGIT, seg)
        + _lookup(CONDITION_LOGIT, cond)
        + _lookup(PASSABILITY, df["circuit"])
        - 0.2 * boost_d * soc_d  # a defender with energy left can fight back on the exit
    )
    sticks = rng.random(n) < 1 / (1 + np.exp(-logit))
    return alongside & sticks


def sample_scenarios(n, rng):
    """Draw `n` varied race situations covering the whole feature space."""
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

    seg = rng.choice(SEGMENTS, n, p=[0.55, 0.20, 0.15, 0.10])
    gap = rng.uniform(0.05, 2.0, n)
    # The gap at the detection point differs a little from the gap when the attack starts.
    detection_gap = gap + rng.normal(0.0, 0.2, n)
    return pd.DataFrame({
        "track_condition": cond,
        "segment": seg,
        "circuit": rng.choice(CIRCUITS, n),
        "attacker_tyre": tyres(),
        "defender_tyre": tyres(),
        "gap_s": gap,
        "run_length_m": np.where(seg == "straight", rng.uniform(250, 1300, n), rng.uniform(50, 300, n)),
        "attacker_battery": rng.uniform(0, 100, n),
        "defender_battery": rng.uniform(0, 100, n),
        "attacker_tyre_age": rng.integers(0, 41, n).astype(float),
        "defender_tyre_age": rng.integers(0, 41, n).astype(float),
        "track_temp_c": np.where(cond == "dry", rng.uniform(18, 55, n), rng.uniform(10, 32, n)),
        "attacker_skill": rng.uniform(60, 100, n),
        "defender_skill": rng.uniform(60, 100, n),
        "top_speed_delta_kph": np.clip(rng.normal(0, 5, n), -15, 15),
        "pace_delta_s": np.clip(rng.normal(0, 0.4, n), -1.5, 1.5),
        "attacker_boost": rng.random(n) < 0.8,
        "defender_boost": rng.random(n) < 0.5,
        "overtake_mode": detection_gap <= 1.0,
    })


def generate_dataset(n, seed=0):
    """Sample `n` scenarios and simulate one attempt each, adding the `overtake` label."""
    rng = np.random.default_rng(seed)
    df = sample_scenarios(n, rng)
    df[LABEL] = simulate(df, rng).astype(int)
    return df


def true_probability(df, n_runs=400, seed=1):
    """Ground-truth probability per row, estimated by replaying it `n_runs` times."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    repeated = df.loc[df.index.repeat(n_runs)].reset_index(drop=True)
    return simulate(repeated, rng).reshape(len(df), n_runs).mean(axis=1)
