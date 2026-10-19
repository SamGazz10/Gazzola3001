"""Alternative models the neural network is compared against.

* **Logistic regression** - a generalised linear model. Interpretable and
  quick, but each input can only shift the log-odds by a fixed amount, so it
  cannot represent interactions such as "battery charge matters more on a
  long straight".
* **Histogram gradient-boosted trees** - an ensemble of decision trees, a
  strong standard choice for tabular data that captures interactions.

Both use exactly the same inputs as the network.
"""
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import BINARY, CATEGORICAL, LABEL, NUMERIC, VOCABS, as_frame, validate

COLUMNS = list(CATEGORICAL) + NUMERIC + BINARY


def _inputs(df):
    df = as_frame(df)
    validate(df)
    return df[COLUMNS].astype({c: int for c in BINARY})


class SklearnModel:
    """A fitted scikit-learn classifier with the same ``predict`` interface as the network.

    Attributes
    ----------
    name : str
    pipeline : sklearn.pipeline.Pipeline
    fit_seconds : float
        Wall-clock training time.
    """

    def __init__(self, name, pipeline):
        self.name = name
        self.pipeline = pipeline
        self.fit_seconds = None

    def fit(self, train_df):
        """Fit on a DataFrame with the schema columns and the ``overtake`` label. Returns self."""
        start = time.perf_counter()
        self.pipeline.fit(_inputs(train_df), train_df[LABEL])
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict(self, scenarios):
        """P(overtake) for a DataFrame of scenarios.

        Returns
        -------
        numpy.ndarray
        """
        return self.pipeline.predict_proba(_inputs(scenarios))[:, 1]


def logistic_regression():
    """Unfitted logistic regression on one-hot categoricals and standardised numerics."""
    columns = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ("numeric", StandardScaler(), NUMERIC),
        ("binary", "passthrough", BINARY),
    ])
    return SklearnModel("Logistic regression", make_pipeline(columns, LogisticRegression(max_iter=2000)))


def gradient_boosting(seed=0):
    """Unfitted histogram gradient-boosted trees with native categorical splits.

    Settings: 500 trees maximum with early stopping on a 10 % internal
    validation split, learning rate 0.1, 31 leaves per tree.
    """
    columns = ColumnTransformer([
        ("categorical", OrdinalEncoder(categories=[VOCABS[v] for v in CATEGORICAL.values()]), list(CATEGORICAL)),
        ("rest", "passthrough", NUMERIC + BINARY),
    ])
    n_cat = len(CATEGORICAL)
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
        validation_fraction=0.1, categorical_features=np.arange(n_cat), random_state=seed,
    )
    return SklearnModel("Gradient boosting", make_pipeline(columns, model))
