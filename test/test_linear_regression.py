import pytest
import numpy as np
from sklearn.datasets import make_regression
from ppyPatternRecognition.regression.linear_regression import LinearRegression

def test_linear_regression_fit():
    # generate data
    X, y = make_regression(n_samples=100, n_features=1, random_state=0)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)

    # Check if coefficients are not None
    assert lr.coef_ is not None
    assert lr.intercept_ is not None

def test_linear_regression_predict():
    # generate data
    X, y = make_regression(n_samples=100, n_features=1, random_state=0)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)

    # Predict
    y_pred = lr.predict(X)

    # Check if predictions are of correct shape
    assert y_pred.shape == y.shape

def test_linear_regression_score():
    # generate data
    X, y = make_regression(n_samples=100, n_features=1, random_state=0)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)

    # Score
    score = lr.score(X, y)

    # Check if score is a valid value
    assert np.isfinite(score)

def test_linear_regression_get_params():
    # Linear Regression
    lr = LinearRegression()

    # Get params
    params = lr.get_params()

    # Check if params is a dictionary
    assert isinstance(params, dict)

def test_linear_regression_set_params():
    # Linear Regression
    lr = LinearRegression()

    # Set params
    params = {'alpha': 0.5}
    lr.set_params(**params)

    # Check if params are set correctly
    assert lr.alpha == 0.5