"""Predict the probability of an overtake for a race situation.

    python -m overtake.predict --condition dry --tyre soft --segment straight --battery 100 --gap 1.0

From Python:

    from overtake.predict import OvertakePredictor
    predictor = OvertakePredictor.load()
    predictor.predict({"track_condition": "dry", "attacker_tyre": "soft", "gap_s": 1.0})

Fields you leave out take the values in `features.DEFAULTS`; `overtake_mode`
defaults to "gap is 1 s or less".
"""
import argparse
from pathlib import Path

import torch

from .features import CIRCUITS, COMPOUNDS, CONDITIONS, DEFAULTS, NUMERIC, SEGMENTS, Preprocessor, as_frame
from .model import OvertakeNet

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "overtake_net.pt"


class OvertakePredictor:
    def __init__(self, model, preprocessor):
        self.model = model.eval()
        self.preprocessor = preprocessor

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path}; run `python -m overtake.train` first")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = OvertakeNet(checkpoint["hidden"], checkpoint["dropout"])
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, Preprocessor.from_state_dict(checkpoint["preprocessor"]))

    def save(self, path, **extra):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "hidden": self.model.hidden,
            "dropout": self.model.dropout,
            "preprocessor": self.preprocessor.state_dict(),
            **extra,
        }, path)

    def predict(self, scenarios):
        """P(overtake) for one scenario (dict -> float) or many (list/DataFrame -> array)."""
        df = complete(scenarios)
        cat, num = self.preprocessor.transform(df)
        with torch.no_grad():
            p = torch.sigmoid(self.model(cat, num)).numpy()
        return float(p[0]) if isinstance(scenarios, dict) else p

    def out_of_range(self, scenario):
        """Inputs outside the range seen in training, where predictions are extrapolation."""
        df = complete(scenario)
        warnings = []
        for col in NUMERIC:
            lo, hi = self.preprocessor.mins[col], self.preprocessor.maxs[col]
            slack = 0.02 * (hi - lo)
            value = float(df[col].iloc[0])
            if not lo - slack <= value <= hi + slack:
                warnings.append(f"{col}={value:g} is outside the training range {lo:g}-{hi:g}")
        return warnings


def complete(scenarios):
    """Fill in missing fields from DEFAULTS."""
    df = as_frame(scenarios).copy()
    for col, value in DEFAULTS.items():
        if col != "overtake_mode":
            df[col] = df[col].fillna(value) if col in df else value
    if "overtake_mode" not in df:
        df["overtake_mode"] = df["gap_s"] <= 1.0
    df["overtake_mode"] = df["overtake_mode"].fillna(df["gap_s"] <= 1.0).astype(bool)
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--condition", choices=CONDITIONS, default=DEFAULTS["track_condition"])
    p.add_argument("--segment", choices=SEGMENTS, default=DEFAULTS["segment"])
    p.add_argument("--circuit", choices=CIRCUITS, default=DEFAULTS["circuit"])
    p.add_argument("--gap", type=float, default=DEFAULTS["gap_s"], help="seconds to the car ahead")
    p.add_argument("--run-length", type=float,
                   help="metres of straight before the braking point (default 800 on a straight, 150 into a corner)")
    p.add_argument("--track-temp", type=float, help="deg C (default 35 dry, 20 damp/wet)")

    a = p.add_argument_group("attacker")
    a.add_argument("--tyre", choices=COMPOUNDS, default=DEFAULTS["attacker_tyre"])
    a.add_argument("--tyre-age", type=float, default=DEFAULTS["attacker_tyre_age"], help="laps")
    a.add_argument("--battery", type=float, default=DEFAULTS["attacker_battery"], help="state of charge, %%")
    a.add_argument("--skill", type=float, default=DEFAULTS["attacker_skill"], help="racecraft rating 0-100")
    a.add_argument("--no-boost", action="store_true", help="attack without Boost")
    a.add_argument("--overtake-mode", choices=["auto", "on", "off"], default="auto",
                   help="auto = on when the gap is 1 s or less")
    a.add_argument("--top-speed-delta", type=float, default=0.0, help="car straight-line advantage, km/h")
    a.add_argument("--pace-delta", type=float, default=0.0, help="car lap-time advantage, s/lap")

    d = p.add_argument_group("defender")
    d.add_argument("--defender-tyre", choices=COMPOUNDS, default=DEFAULTS["defender_tyre"])
    d.add_argument("--defender-tyre-age", type=float, default=DEFAULTS["defender_tyre_age"])
    d.add_argument("--defender-battery", type=float, default=DEFAULTS["defender_battery"])
    d.add_argument("--defender-skill", type=float, default=DEFAULTS["defender_skill"])
    d.add_argument("--defender-no-boost", action="store_true", help="defender does not use Boost")

    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return p.parse_args(argv)


def scenario_from_args(args):
    if args.run_length is None:
        args.run_length = 800.0 if args.segment == "straight" else 150.0
    if args.track_temp is None:
        args.track_temp = 35.0 if args.condition == "dry" else 20.0
    return {
        "track_condition": args.condition,
        "segment": args.segment,
        "circuit": args.circuit,
        "attacker_tyre": args.tyre,
        "defender_tyre": args.defender_tyre,
        "gap_s": args.gap,
        "run_length_m": args.run_length,
        "attacker_battery": args.battery,
        "defender_battery": args.defender_battery,
        "attacker_tyre_age": args.tyre_age,
        "defender_tyre_age": args.defender_tyre_age,
        "track_temp_c": args.track_temp,
        "attacker_skill": args.skill,
        "defender_skill": args.defender_skill,
        "top_speed_delta_kph": args.top_speed_delta,
        "pace_delta_s": args.pace_delta,
        "attacker_boost": not args.no_boost,
        "defender_boost": not args.defender_no_boost,
        "overtake_mode": {"auto": args.gap <= 1.0, "on": True, "off": False}[args.overtake_mode],
    }


def describe(s):
    on = {True: "on", False: "off"}
    return "\n".join([
        f"  Track:    {s['track_condition']}, {s['track_temp_c']:g} C, {s['circuit']}, "
        f"{s['segment'].replace('_', ' ')} ({s['run_length_m']:g} m run)",
        f"  Attacker: {s['attacker_tyre']} ({s['attacker_tyre_age']:g} laps), battery {s['attacker_battery']:g}%, "
        f"Boost {on[s['attacker_boost']]}, Overtake Mode {on[s['overtake_mode']]}, skill {s['attacker_skill']:g}",
        f"  Defender: {s['defender_tyre']} ({s['defender_tyre_age']:g} laps), battery {s['defender_battery']:g}%, "
        f"Boost {on[s['defender_boost']]}, skill {s['defender_skill']:g}",
        f"  Gap:      {s['gap_s']:.2f} s",
    ])


def main(argv=None):
    args = parse_args(argv)
    predictor = OvertakePredictor.load(args.model)
    scenario = scenario_from_args(args)
    p = predictor.predict(scenario)

    print("Scenario")
    print(describe(scenario))
    print(f"\nP(overtake) = {p:.1%}")
    if scenario["attacker_boost"]:
        p_no_boost = predictor.predict({**scenario, "attacker_boost": False})
        print(f"  without Boost: {p_no_boost:.1%}  (Boost adds {100 * (p - p_no_boost):+.1f} points)")
    for warning in predictor.out_of_range(scenario):
        print(f"  warning: {warning}")

    print("\nSame situation at other gaps")
    for gap in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        q = predictor.predict({**scenario, "gap_s": gap,
                               "overtake_mode": gap <= 1.0 if args.overtake_mode == "auto" else scenario["overtake_mode"]})
        print(f"  {gap:4.2f} s  {q:6.1%}")


if __name__ == "__main__":
    main()
