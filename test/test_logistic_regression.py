import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ppyPatternRecognition.regression.logistic_regression import LogisticRegression

def test_logistic_regression_fit():
    # generate data
    X, y = make_classification(n_samples=1000, n_features=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # check if coefficients are not None
    assert lr.coef_ is not None

def test_logistic_regression_predict():
    # generate data
    X, y = make_classification(n_samples=1000, n_features=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # predict on test data
    y_pred = lr.predict(X_test)

    # check if predictions are of correct shape
    assert y_pred.shape == y_test.shape

def test_logistic_regression_score():
    # generate data
    X, y = make_classification(n_samples=1000, n_features=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # calculate accuracy score
    score = lr.score(X_test, y_test)

    # check if score is between 0 and 1
    assert 0 <= score <= 1

def test_logistic_regression_get_params():
    # Logistic Regression
    lr = LogisticRegression()

    # get parameters
    params = lr.get_params()

    # check if parameters are not None
    assert params is not None

def test_logistic_regression_set_params():
    # Logistic Regression
    lr = LogisticRegression()

    # set parameters
    params = {'C': 0.1, 'max_iter': 100}
    lr.set_params(**params)

    # check if parameters are set correctly
    assert lr.C == 0.1
    assert lr.max_iter == 100