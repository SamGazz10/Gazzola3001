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
    return split(generate_dataset(5_000, seed=3), seed=3)


@pytest.fixture(scope="module")
def predictor(data):
    train_df, val_df, _ = data
    model, history = train_network(train_df, val_df, hidden=(16,), epochs=2, log=lambda _: None)
    assert len(history) == 2
    return model


def test_split_is_80_10_10_and_disjoint(data):
    train_df, val_df, test_df = data
    assert (len(train_df), len(val_df), len(test_df)) == (4000, 500, 500)
    assert not set(train_df.index) & set(val_df.index) | set(train_df.index) & set(test_df.index)


def test_predict_single_and_batch(predictor):
    p = predictor.predict(DEFAULTS)
    assert isinstance(p, float) and 0 <= p <= 1
    batch = predictor.predict([DEFAULTS, {**DEFAULTS, "gap_s": 0.4}])
    assert batch.shape == (2,)


def test_missing_fields_use_defaults(predictor):
    assert predictor.predict({"gap_s": 1.0}) == pytest.approx(predictor.predict(DEFAULTS))


def test_save_and_load_give_same_predictions(predictor, tmp_path):
    path = tmp_path / "model.pt"
    predictor.save(path)
    loaded = OvertakePredictor.load(path)
    scenarios = [{**DEFAULTS, "gap_s": g} for g in np.linspace(0.1, 2.0, 10)]
    np.testing.assert_allclose(loaded.predict(scenarios), predictor.predict(scenarios))


def test_load_missing_checkpoint_explains_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="overtake.train"):
        OvertakePredictor.load(tmp_path / "missing.pt")


@pytest.mark.parametrize("column, value", [
    ("attacker_tyre", "supersoft"),
    ("gap_s", -0.5),
    ("gap_s", "one second"),
    ("run_length_m", 0.0),
    ("attacker_battery", 120.0),
    ("attacker_boost", "no"),
])
def test_invalid_inputs_are_rejected_with_the_column_named(predictor, column, value):
    with pytest.raises(ValueError, match=column):
        predictor.predict({**DEFAULTS, column: value})


def test_none_means_use_the_default(predictor):
    assert predictor.predict({**DEFAULTS, "track_condition": None}) == pytest.approx(predictor.predict(DEFAULTS))


def test_validate_rejects_missing_values():
    with pytest.raises(ValueError, match="track_condition"):
        validate(pd.DataFrame([{**DEFAULTS, "track_condition": None}]))


def test_missing_column_is_reported():
    with pytest.raises(ValueError, match="missing columns"):
        validate(pd.DataFrame([DEFAULTS]).drop(columns="circuit"))


def test_extrapolation_is_flagged(predictor):
    assert predictor.out_of_range(DEFAULTS) == []
    warnings = predictor.out_of_range({**DEFAULTS, "run_length_m": 1900.0})
    assert len(warnings) == 1 and "run_length_m" in warnings[0]


def test_command_line_prints_probability(capsys, predictor, tmp_path):
    path = tmp_path / "model.pt"
    predictor.save(path)
    predict_module.main(["--gap", "0.8", "--battery", "90", "--model", str(path)])
    out = capsys.readouterr().out
    assert "P(overtake) =" in out and "without Boost" in out


def test_command_line_rejects_bad_input(capsys, predictor, tmp_path):
    path = tmp_path / "model.pt"
    predictor.save(path)
    with pytest.raises(SystemExit) as exit_info:
        predict_module.main(["--gap", "5", "--model", str(path)])
    assert exit_info.value.code == 2
    assert "gap_s" in capsys.readouterr().err


def test_corner_run_length_defaults_to_150_m(predictor):
    corner = {"segment": "slow_corner"}
    assert predictor.predict(corner) == pytest.approx(predictor.predict({**corner, "run_length_m": 150.0}))


def test_wet_track_temperature_defaults_to_20_c():
    from overtake.predict import complete
    assert complete({"track_condition": "wet"})["track_temp_c"].iloc[0] == 20.0
    assert complete({"track_condition": "dry"})["track_temp_c"].iloc[0] == DEFAULTS["track_temp_c"]


def test_series_is_one_scenario_and_returns_a_float(predictor):
    scenario = {"gap_s": 0.6, "attacker_battery": 70.0}
    p = predictor.predict(pd.Series(scenario))
    assert isinstance(p, float) and p == pytest.approx(predictor.predict(scenario))


def test_unknown_field_is_rejected(predictor):
    with pytest.raises(ValueError, match="unknown field"):
        predictor.predict({"gap": 1.0})


def test_nan_is_rejected_not_filled(predictor):
    with pytest.raises(ValueError, match="gap_s"):
        predictor.predict({"gap_s": float("nan")})


@pytest.mark.parametrize("scenario, field", [
    ({"segment": "slow_corner", "run_length_m": 800.0}, "run_length_m"),
    ({"track_condition": "wet", "track_temp_c": 45.0}, "track_temp_c"),
    ({"gap_s": 1.9, "overtake_mode": True}, "overtake_mode"),
])
def test_situational_extrapolation_is_flagged(predictor, scenario, field):
    warnings = predictor.out_of_range(scenario)
    assert len(warnings) == 1 and field in warnings[0]


def test_command_line_rejects_nan(capsys):
    with pytest.raises(SystemExit) as exit_info:
        predict_module.main(["--gap", "nan"])
    assert exit_info.value.code == 2
    assert "not a finite number" in capsys.readouterr().err


def test_command_line_missing_model_exits_with_message(capsys, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        predict_module.main(["--model", str(tmp_path / "missing.pt")])
    assert exit_info.value.code == 1
    assert "overtake.train" in capsys.readouterr().err


@pytest.mark.parametrize("physics, width", [(True, 16), (False, 14)])
def test_preprocessor_shapes_and_standardisation(data, physics, width):
    train_df = data[0]
    pre = Preprocessor(physics_features=physics).fit(train_df)
    cat, num = pre.transform(train_df)
    assert cat.shape == (len(train_df), 5) and num.shape == (len(train_df), width) == (len(train_df), pre.n_inputs)
    assert num[:, :width - 3].mean(dim=0).abs().max() < 1e-4  # numeric columns centred on the training set


def test_energy_cover_saturates_once_the_battery_covers_the_run():
    from overtake.features import derived_features
    df = pd.DataFrame([{**DEFAULTS, "attacker_battery": b, "run_length_m": 800.0} for b in (0, 50, 100)])
    cover = derived_features(df)["attacker_energy_cover"]
    assert cover[0] == pytest.approx(0.5e6 / (350e3 * 800 / 80))  # Overtake Mode energy only
    assert cover[0] < cover[1] <= cover[2] == 1.0


@pytest.mark.parametrize("make", [logistic_regression, gradient_boosting])
def test_baselines_fit_and_predict_probabilities(data, make):
    train_df, _, test_df = data
    p = make().fit(train_df).predict(test_df)
    assert p.shape == (len(test_df),) and np.all((p >= 0) & (p <= 1))


def test_metrics_of_perfect_and_uninformative_predictions():
    y = np.array([0, 1, 0, 1])
    perfect = metrics(y, y.astype(float), p_true=y.astype(float))
    assert perfect["accuracy"] == 1 and perfect["roc_auc"] == 1
    assert perfect["mae_vs_true"] == pytest.approx(0, abs=1e-6)  # probabilities are clipped to [1e-7, 1 - 1e-7]
    assert metrics(y, np.full(4, 0.5))["log_loss"] == pytest.approx(np.log(2))


def test_ensemble_averages_its_members_and_round_trips(data, tmp_path):
    train_df, val_df, _ = data
    from overtake.train import train_ensemble
    ensemble, histories = train_ensemble(train_df, val_df, n_members=2, hidden=(8,), epochs=1, log=lambda _: None)
    assert len(ensemble.models) == len(histories) == 2
    scenarios = [{**DEFAULTS, "gap_s": g} for g in (0.3, 1.0, 1.7)]
    members = [OvertakePredictor(m, ensemble.preprocessor).predict(scenarios) for m in ensemble.models]
    np.testing.assert_allclose(ensemble.predict(scenarios), np.mean(members, axis=0), rtol=1e-6)
    ensemble.save(tmp_path / "ensemble.pt")
    np.testing.assert_allclose(OvertakePredictor.load(tmp_path / "ensemble.pt").predict(scenarios),
                               ensemble.predict(scenarios), rtol=1e-6)
