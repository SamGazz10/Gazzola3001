"""Checks on preprocessing, training, saving/loading, prediction and input handling."""
import numpy as np
import pandas as pd
import pytest

from overtake import predict as predict_module
from overtake.baselines import gradient_boosting, logistic_regression
from overtake.features import DEFAULTS, Preprocessor, validate
from overtake.predict import OvertakePredictor
from overtake.simulator import generate_dataset
from overtake.train import metrics, split, train_network


@pytest.fixture(scope="module")
def data():
    """Training, validation and test sets from 5,000 simulated attempts, shared by this module's tests."""
    return split(generate_dataset(5_000, seed=3), seed=3)


@pytest.fixture(scope="module")
def predictor(data):
    """A one-layer, 16-unit network trained for two epochs: enough to exercise the API, not to be accurate."""
    train_df, val_df, _ = data
    model, history = train_network(train_df, val_df, hidden=(16,), epochs=2, log=lambda _: None)
    assert len(history) == 2  # early stopping (patience 5) cannot end a two-epoch run
    return model


def test_split_is_80_10_10_and_disjoint(data):
    """The split is 80/10/10 and no training row reappears in validation or test, which would flatter the scores."""
    train_df, val_df, test_df = data
    assert (len(train_df), len(val_df), len(test_df)) == (4000, 500, 500)
    # & binds tighter than |: this is (train & val) | (train & test).
    assert not set(train_df.index) & set(val_df.index) | set(train_df.index) & set(test_df.index)


def test_predict_single_and_batch(predictor):
    """A dict gives one float probability; a list gives an array with one probability per scenario."""
    p = predictor.predict(DEFAULTS)
    assert isinstance(p, float) and 0 <= p <= 1
    batch = predictor.predict([DEFAULTS, {**DEFAULTS, "gap_s": 0.4}])
    assert batch.shape == (2,)


def test_missing_fields_use_defaults(predictor):
    """Omitted fields take `DEFAULTS` (Overtake Mode on, as the gap is 1 s), so a partial scenario is the default one."""
    assert predictor.predict({"gap_s": 1.0}) == pytest.approx(predictor.predict(DEFAULTS))


def test_save_and_load_give_same_predictions(predictor, tmp_path):
    """A saved and reloaded checkpoint predicts as the original, so a shipped model reproduces the trained one."""
    path = tmp_path / "model.pt"
    predictor.save(path)
    loaded = OvertakePredictor.load(path)
    scenarios = [{**DEFAULTS, "gap_s": g} for g in np.linspace(0.1, 2.0, 10)]
    np.testing.assert_allclose(loaded.predict(scenarios), predictor.predict(scenarios))


def test_load_missing_checkpoint_explains_how_to_train(tmp_path):
    """Loading a missing checkpoint raises FileNotFoundError telling the user to run ``python -m overtake.train``."""
    with pytest.raises(FileNotFoundError, match="overtake.train"):
        OvertakePredictor.load(tmp_path / "missing.pt")


@pytest.mark.parametrize("column, value", [
    ("attacker_tyre", "supersoft"),   # unknown category
    ("gap_s", -0.5),                  # below the 0 s limit
    ("gap_s", "one second"),          # not a number
    ("run_length_m", 0.0),            # below the 10 m limit
    ("attacker_battery", 120.0),      # above 100 %
    ("attacker_boost", "no"),         # not a boolean
])
def test_invalid_inputs_are_rejected_with_the_column_named(predictor, column, value):
    """Each kind of invalid input raises ValueError naming the column, so the user knows what to fix."""
    with pytest.raises(ValueError, match=column):
        predictor.predict({**DEFAULTS, column: value})


def test_none_means_use_the_default(predictor):
    """In the prediction API a field set to None takes its default, as if it were omitted."""
    assert predictor.predict({**DEFAULTS, "track_condition": None}) == pytest.approx(predictor.predict(DEFAULTS))


def test_validate_rejects_missing_values():
    """`validate` itself fills nothing in: a None category is rejected, so it can never reach the network."""
    with pytest.raises(ValueError, match="track_condition"):
        validate(pd.DataFrame([{**DEFAULTS, "track_condition": None}]))


def test_missing_column_is_reported():
    """A table without a schema column is rejected with the missing column listed."""
    with pytest.raises(ValueError, match="missing columns"):
        validate(pd.DataFrame([DEFAULTS]).drop(columns="circuit"))


def test_extrapolation_is_flagged(predictor):
    """A 1,900 m straight is accepted (limit 2,000 m) but lies beyond the 1,300 m trained, so it gets one warning."""
    assert predictor.out_of_range(DEFAULTS) == []
    warnings = predictor.out_of_range({**DEFAULTS, "run_length_m": 1900.0})
    assert len(warnings) == 1 and "run_length_m" in warnings[0]


def test_command_line_prints_probability(capsys, predictor, tmp_path):
    """``python -m overtake.predict`` prints the probability and the same attack without Boost."""
    path = tmp_path / "model.pt"
    predictor.save(path)
    predict_module.main(["--gap", "0.8", "--battery", "90", "--model", str(path)])
    out = capsys.readouterr().out
    assert "P(overtake) =" in out and "without Boost" in out


def test_command_line_rejects_bad_input(capsys, predictor, tmp_path):
    """A 5 s gap (limit 3 s) makes the command line exit with status 2 and name ``gap_s`` on stderr."""
    path = tmp_path / "model.pt"
    predictor.save(path)
    with pytest.raises(SystemExit) as exit_info:
        predict_module.main(["--gap", "5", "--model", str(path)])
    assert exit_info.value.code == 2
    assert "gap_s" in capsys.readouterr().err


def test_corner_run_length_defaults_to_150_m(predictor):
    """A move into a corner without a run length uses 150 m, not the 800 m straight of `DEFAULTS`."""
    corner = {"segment": "slow_corner"}
    assert predictor.predict(corner) == pytest.approx(predictor.predict({**corner, "run_length_m": 150.0}))


def test_wet_track_temperature_defaults_to_20_c():
    """A wet track without a temperature uses 20 C; a dry one keeps the default 35 C."""
    from overtake.predict import complete
    assert complete({"track_condition": "wet"})["track_temp_c"].iloc[0] == 20.0
    assert complete({"track_condition": "dry"})["track_temp_c"].iloc[0] == DEFAULTS["track_temp_c"]


def test_series_is_one_scenario_and_returns_a_float(predictor):
    """A pandas Series (such as one DataFrame row) is one scenario and gives a float, like a dict."""
    scenario = {"gap_s": 0.6, "attacker_battery": 70.0}
    p = predictor.predict(pd.Series(scenario))
    assert isinstance(p, float) and p == pytest.approx(predictor.predict(scenario))


def test_unknown_field_is_rejected(predictor):
    """A misspelt field ("gap" for "gap_s") raises instead of being ignored and replaced by the default."""
    with pytest.raises(ValueError, match="unknown field"):
        predictor.predict({"gap": 1.0})


def test_nan_is_rejected_not_filled(predictor):
    """NaN is an invalid value, not a missing one: it raises instead of taking the default."""
    with pytest.raises(ValueError, match="gap_s"):
        predictor.predict({"gap_s": float("nan")})


@pytest.mark.parametrize("scenario, field", [
    ({"segment": "slow_corner", "run_length_m": 800.0}, "run_length_m"),  # corners trained on 50-300 m
    ({"track_condition": "wet", "track_temp_c": 45.0}, "track_temp_c"),   # wet trained on 10-32 C
    ({"gap_s": 1.9, "overtake_mode": True}, "overtake_mode"),             # on only up to a 1.6 s gap in training
])
def test_situational_extrapolation_is_flagged(predictor, scenario, field):
    """Inputs inside the global training range but unseen in their situation get exactly one warning."""
    warnings = predictor.out_of_range(scenario)
    assert len(warnings) == 1 and field in warnings[0]


def test_command_line_rejects_nan(capsys):
    """``--gap nan`` is rejected by argparse (exit status 2) before any model is loaded."""
    with pytest.raises(SystemExit) as exit_info:
        predict_module.main(["--gap", "nan"])
    assert exit_info.value.code == 2
    assert "not a finite number" in capsys.readouterr().err


def test_command_line_missing_model_exits_with_message(capsys, tmp_path):
    """A missing model file makes the command line exit with status 1 and say how to train one."""
    with pytest.raises(SystemExit) as exit_info:
        predict_module.main(["--model", str(tmp_path / "missing.pt")])
    assert exit_info.value.code == 1
    assert "overtake.train" in capsys.readouterr().err


@pytest.mark.parametrize("physics, width", [(True, 16), (False, 14)])
def test_preprocessor_shapes_and_standardisation(data, physics, width):
    """5 category indices and 11 numeric + 3 flag inputs (+2 energy-cover), numeric ones centred on training data."""
    train_df = data[0]
    pre = Preprocessor(physics_features=physics).fit(train_df)
    cat, num = pre.transform(train_df)
    assert cat.shape == (len(train_df), 5) and num.shape == (len(train_df), width) == (len(train_df), pre.n_inputs)
    # The last 3 columns are the 0/1 flags, which are not standardised; 1e-4 allows for float32 round-off.
    assert num[:, :width - 3].mean(dim=0).abs().max() < 1e-4  # numeric columns centred on the training set


def test_energy_cover_saturates_once_the_battery_covers_the_run():
    """Energy cover rises with battery charge and is capped at 1 once the battery can power the whole run."""
    from overtake.features import derived_features
    df = pd.DataFrame([{**DEFAULTS, "attacker_battery": b, "run_length_m": 800.0} for b in (0, 50, 100)])
    cover = derived_features(df)["attacker_energy_cover"]
    # 350 kW for 800 m at 80 m/s needs 3.5 MJ; an empty battery leaves the 0.5 MJ of Overtake Mode.
    assert cover[0] == pytest.approx(0.5e6 / (350e3 * 800 / 80))  # Overtake Mode energy only
    assert cover[0] < cover[1] <= cover[2] == 1.0


@pytest.mark.parametrize("make", [logistic_regression, gradient_boosting])
def test_baselines_fit_and_predict_probabilities(data, make):
    """Both baselines fit the training set and return one probability in [0, 1] per test attempt."""
    train_df, _, test_df = data
    p = make().fit(train_df).predict(test_df)
    assert p.shape == (len(test_df),) and np.all((p >= 0) & (p <= 1))


def test_metrics_of_perfect_and_uninformative_predictions():
    """Perfect predictions score 1 for accuracy and ROC AUC; a constant 0.5 has log loss ln 2."""
    y = np.array([0, 1, 0, 1])
    perfect = metrics(y, y.astype(float), p_true=y.astype(float))
    assert perfect["accuracy"] == 1 and perfect["roc_auc"] == 1
    assert perfect["mae_vs_true"] == pytest.approx(0, abs=1e-6)  # probabilities are clipped to [1e-7, 1 - 1e-7]
    # -ln(0.5) = ln 2 for every attempt, whatever the outcome.
    assert metrics(y, np.full(4, 0.5))["log_loss"] == pytest.approx(np.log(2))


def test_ensemble_averages_its_members_and_round_trips(data, tmp_path):
    """An ensemble predicts the mean of its members' probabilities, and keeps doing so after saving and loading."""
    train_df, val_df, _ = data
    from overtake.train import train_ensemble
    ensemble, histories = train_ensemble(train_df, val_df, n_members=2, hidden=(8,), epochs=1, log=lambda _: None)
    assert len(ensemble.models) == len(histories) == 2
    scenarios = [{**DEFAULTS, "gap_s": g} for g in (0.3, 1.0, 1.7)]
    members = [OvertakePredictor(m, ensemble.preprocessor).predict(scenarios) for m in ensemble.models]
    # rtol 1e-6: the library averages in float32.
    np.testing.assert_allclose(ensemble.predict(scenarios), np.mean(members, axis=0), rtol=1e-6)
    ensemble.save(tmp_path / "ensemble.pt")
    np.testing.assert_allclose(OvertakePredictor.load(tmp_path / "ensemble.pt").predict(scenarios),
                               ensemble.predict(scenarios), rtol=1e-6)
