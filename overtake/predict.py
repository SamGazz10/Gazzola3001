"""Predict the probability of an overtake for a race situation.

    python -m overtake.predict --condition dry --tyre soft --segment straight --battery 100 --gap 1.0

From Python:

    from overtake.predict import OvertakePredictor
    predictor = OvertakePredictor.load()
    predictor.predict({"track_condition": "dry", "attacker_tyre": "soft", "gap_s": 1.0})

Fields you leave out take the values in `overtake.features.DEFAULTS`;
``overtake_mode`` defaults to "gap is 1 s or less". Invalid inputs raise a
``ValueError`` (the command line prints the message and exits with status 2).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from .features import CIRCUITS, COMPOUNDS, CONDITIONS, DEFAULTS, NUMERIC, SEGMENTS, Preprocessor, as_frame
from .model import OvertakeNet

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "overtake_net.pt"


class OvertakePredictor:
    """One or more trained networks together with the preprocessing they were trained with.

    With several networks (an ensemble trained from different random seeds)
    the prediction is the mean of their probabilities.

    Parameters
    ----------
    models : overtake.model.OvertakeNet or list of them
        All trained on the same data, so they share one preprocessor.
    preprocessor : overtake.features.Preprocessor
        Fitted on the networks' training data.
    """

    def __init__(self, models, preprocessor):
        self.models = [m.eval() for m in (models if isinstance(models, (list, tuple)) else [models])]
        self.preprocessor = preprocessor

    @property
    def model(self):
        """The first network (all members share the same architecture)."""
        return self.models[0]

    @property
    def n_parameters(self):
        """Trainable parameters across all networks."""
        return sum(p.numel() for m in self.models for p in m.parameters())

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        """Load a checkpoint written by `save`.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Defaults to ``models/overtake_net.pt``.

        Returns
        -------
        OvertakePredictor

        Raises
        ------
        FileNotFoundError
            If there is no checkpoint at ``path``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path}; run `python -m overtake.train` first")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        models = []
        for state in checkpoint["model_states"]:
            model = OvertakeNet(checkpoint["hidden"], checkpoint["dropout"], checkpoint["n_inputs"])
            model.load_state_dict(state)
            models.append(model)
        return cls(models, Preprocessor.from_state_dict(checkpoint["preprocessor"]))

    def save(self, path, **extra):
        """Write weights, architecture and preprocessing statistics to ``path``.

        Extra keyword arguments (e.g. metrics, training history) are stored
        alongside and must be plain Python values.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_states": [m.state_dict() for m in self.models],
            "hidden": self.model.hidden,
            "dropout": self.model.dropout,
            "n_inputs": self.model.n_inputs,
            "preprocessor": self.preprocessor.state_dict(),
            **extra,
        }, path)

    def predict(self, scenarios):
        """Probability of an overtake.

        Parameters
        ----------
        scenarios : dict, list of dict or pandas.DataFrame
            One or more scenarios in the `overtake.features` schema. Missing
            fields take their default values.

        Returns
        -------
        float or numpy.ndarray
            A float for a single dict, otherwise one probability per row.

        Raises
        ------
        ValueError
            If a value is outside the schema (see `overtake.features.validate`).
        """
        df = complete(scenarios)
        cat, num = self.preprocessor.transform(df)
        with torch.no_grad():
            p = torch.stack([torch.sigmoid(m(cat, num)) for m in self.models]).mean(dim=0).numpy()
        return float(p[0]) if isinstance(scenarios, dict) else p

    def out_of_range(self, scenario):
        """List inputs outside the range seen in training, where predictions are extrapolation.

        Parameters
        ----------
        scenario : dict
            One scenario (missing fields take default values).

        Returns
        -------
        list of str
            One message per numeric input more than 2 % of its range outside
            the training data's minimum-maximum range; empty if none.
        """
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
    """Return the scenarios as a DataFrame with missing fields filled from `DEFAULTS`."""
    df = as_frame(scenarios).copy()
    for col, value in DEFAULTS.items():
        if col != "overtake_mode":
            df[col] = df[col].fillna(value) if col in df else value
    # Non-numeric gaps are left for validate() to reject with a clear message.
    within_one_second = pd.to_numeric(df["gap_s"], errors="coerce") <= 1.0
    if "overtake_mode" not in df:
        df["overtake_mode"] = within_one_second
    df["overtake_mode"] = df["overtake_mode"].where(df["overtake_mode"].notna(), within_one_second)
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
    """Command-line entry point; see ``python -m overtake.predict --help``."""
    args = parse_args(argv)
    predictor = OvertakePredictor.load(args.model)
    scenario = scenario_from_args(args)
    try:
        p = predictor.predict(scenario)
    except ValueError as error:
        print(f"invalid input: {error}", file=sys.stderr)
        raise SystemExit(2)

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
