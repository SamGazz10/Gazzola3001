"""Alternative models the neural network is compared against.

* Logistic regression, a generalised linear model. Interpretable and quick,
  but each input can only shift the log-odds by a fixed amount, so it cannot
  represent interactions such as "battery charge matters more on a long
  straight".
* Histogram gradient-boosted trees, an ensemble of decision trees: a strong
  standard choice for tabular data that captures interactions.

Both use the same 19 raw inputs as the final network
(`overtake.features.FEATURES`, without the `overtake.features.DERIVED`
columns) and reject invalid input with the same `overtake.features.validate`.
"""
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import BINARY, CATEGORICAL, LABEL, NUMERIC, VOCABS, as_frame, validate

COLUMNS = list(CATEGORICAL) + NUMERIC + BINARY  # same columns and order as `overtake.features.FEATURES`


def _inputs(df):
    """Validate scenarios and return the schema columns as scikit-learn input (raises ValueError if invalid)."""
    df = as_frame(df)
    validate(df)
    # Flags may arrive as bool or 1/0 (even mixed); cast so they reach scikit-learn as integers.
    return df[COLUMNS].astype({c: int for c in BINARY})


class SklearnModel:
    """A scikit-learn classifier behind the same ``predict`` call as the network.

    `logistic_regression` and `gradient_boosting` build unfitted instances.
    Unlike `overtake.predict.OvertakePredictor.predict`, ``predict`` here always
    returns an array, even for a single scenario.

    Parameters
    ----------
    name : str
        Label used in results tables and figures.
    pipeline : sklearn.pipeline.Pipeline
        Column preprocessing followed by a classifier with ``predict_proba``.

    Attributes
    ----------
    name : str
    pipeline : sklearn.pipeline.Pipeline
    fit_seconds : float or None
        Wall-clock time of the last `fit` in seconds, including input
        validation; ``None`` until fitted.

    Examples
    --------
    >>> from overtake import simulator
    >>> from overtake.features import DEFAULTS
    >>> data = simulator.generate_dataset(2000, seed=0)
    >>> model = logistic_regression().fit(data)
    >>> model.predict(data).shape, model.predict(DEFAULTS).shape
    ((2000,), (1,))
    """

    def __init__(self, name, pipeline):
        """Wrap an unfitted pipeline; see the class docstring."""
        self.name = name
        self.pipeline = pipeline
        self.fit_seconds = None

    def fit(self, train_df):
        """Fit the pipeline and record how long it took.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training scenarios in the `overtake.features` schema plus the 0/1
            ``overtake`` label column.

        Returns
        -------
        SklearnModel
            ``self``, now fitted, so calls can be chained.

        Raises
        ------
        ValueError
            If the scenarios fail `overtake.features.validate`.
        KeyError
            If the ``overtake`` label column is missing.
        """
        start = time.perf_counter()
        # The inputs get a fresh index from as_frame; scikit-learn pairs rows
        # with labels by position, so the label's original index does not matter.
        self.pipeline.fit(_inputs(train_df), train_df[LABEL])
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict(self, scenarios):
        """P(overtake) for one or more scenarios.

        Parameters
        ----------
        scenarios : dict, pandas.Series, list of dict or pandas.DataFrame
            Complete scenarios in the `overtake.features` schema (anything
            `overtake.features.as_frame` accepts). Missing fields are not
            filled with defaults.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Probability of an overtake for each scenario.

        Raises
        ------
        ValueError
            If the scenarios fail `overtake.features.validate`.
        """
        # Column 1 is the probability of class 1 (classes are sorted, so 0 then 1).
        return self.pipeline.predict_proba(_inputs(scenarios))[:, 1]


def logistic_regression(C=1.0):
    """Unfitted logistic regression on one-hot categoricals and standardised numerics.

    Parameters
    ----------
    C : float, optional
        Inverse strength of the L2 penalty (scikit-learn's default, 1, is used
        in the comparison; `overtake.experiments.tune_baselines` tries others).

    Returns
    -------
    SklearnModel
        Named ``"Logistic regression"``; call `SklearnModel.fit` before predicting.
    """
    columns = ColumnTransformer([
        # A vocabulary value that is absent from the training set encodes as all zeros instead of raising.
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ("numeric", StandardScaler(), NUMERIC),
        ("binary", "passthrough", BINARY),
    ])
    # max_iter: headroom over scikit-learn's default of 100 for the lbfgs solver to converge (estimate).
    return SklearnModel("Logistic regression", make_pipeline(columns, LogisticRegression(C=C, max_iter=2000)))


def gradient_boosting(seed=0, learning_rate=0.1, max_leaf_nodes=31, max_iter=500):
    """Unfitted histogram gradient-boosted trees with native categorical splits.

    The defaults are the settings used in the comparison: at most 500 trees,
    learning rate 0.1, 31 leaves per tree (`overtake.experiments.tune_baselines`
    tries others). Training stops early once the loss on a stratified 10 %
    internal validation split has not improved for 10 iterations
    (scikit-learn's default ``n_iter_no_change``).

    Parameters
    ----------
    seed : int, optional
        Seeds the internal validation split and, when more than 200,000 rows
        remain after it (as with the 240,000-row training set), the subsample
        scikit-learn uses to choose bin edges.
    learning_rate : float, optional
        Shrinkage applied to each tree's contribution.
    max_leaf_nodes : int, optional
        Maximum leaves per tree.
    max_iter : int, optional
        Maximum number of trees (one per boosting iteration for a binary label).

    Returns
    -------
    SklearnModel
        Named ``"Gradient boosting"``; call `SklearnModel.fit` before predicting.
    """
    columns = ColumnTransformer([
        # Fixed vocabularies give each value the same code whatever the training set contains.
        ("categorical", OrdinalEncoder(categories=[VOCABS[v] for v in CATEGORICAL.values()]), list(CATEGORICAL)),
        ("rest", "passthrough", NUMERIC + BINARY),
    ])
    # The transformer outputs its blocks in the order listed, so the n_cat
    # ordinal-coded columns come first and are the ones treated as categorical.
    n_cat = len(CATEGORICAL)
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, early_stopping=True,
        validation_fraction=0.1, categorical_features=np.arange(n_cat), random_state=seed,
    )
    return SklearnModel("Gradient boosting", make_pipeline(columns, model))
