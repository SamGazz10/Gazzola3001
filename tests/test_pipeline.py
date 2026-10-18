import numpy as np
import pytest

from overtake.features import DEFAULTS, Preprocessor
from overtake.predict import OvertakePredictor
from overtake.simulator import generate_dataset
from overtake.train import split, train_network


@pytest.fixture(scope="module")
def predictor():
    train_df, val_df, _ = split(generate_dataset(5_000, seed=3), seed=3)
    model, history = train_network(train_df, val_df, epochs=2, log=lambda _: None)
    assert len(history) == 2
    return model


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


def test_rejects_unknown_tyre(predictor):
    with pytest.raises(ValueError, match="attacker_tyre"):
        predictor.predict({**DEFAULTS, "attacker_tyre": "supersoft"})


def test_preprocessor_shapes():
    df = generate_dataset(100)
    cat, num = Preprocessor().fit(df).transform(df)
    assert cat.shape == (100, 5) and num.shape == (100, 14)
