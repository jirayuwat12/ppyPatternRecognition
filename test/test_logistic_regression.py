from math import e
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys

from sympy import false
sys.path.append('../ppyPatternRecognition')
from ppyPatternRecognition.regression.logistic_regression import LogisticRegression
from ppyPatternRecognition.evaluation import accuracy

# Test LogisticRegression class
class TestLogisticRegression:
    def test_predict(self):
        # Create a LogisticRegression instance
        lr = LogisticRegression(in_features=10)

        # Generate random data for testing
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model on the training data
        lr.fit(X_train, y_train)

        # Predict the output for the test data
        y_pred = lr.predict(X_test)
        y_pred_pb = lr.predict_proba(X_test)

        # Check if the predicted output has the correct shape
        assert y_pred.shape == (X_test.shape[0],)

        # Check if the predicted output contains only 0s and 1s
        assert np.all(np.logical_or(y_pred == 0, y_pred == 1))

        # Check if the predicted output is within the range [0, 1]
        assert np.all(np.logical_and(y_pred >= 0, y_pred <= 1))

        # Check if the predicted output is the same as the predicted probability
        assert np.all(y_pred == (y_pred_pb > 0.5))

    def test_predict_proba(self):
        # Create a LogisticRegression instance
        lr = LogisticRegression(in_features=10)

        # Generate random data for testing
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model on the training data
        lr.fit(X_train, y_train)

        # Predict the output for the test data
        y_pred_pb = lr.predict_proba(X_test)

        # Check if the predicted probability has the correct shape
        assert y_pred_pb.shape == (X_test.shape[0],)

        # Check if the predicted probability is within the range [0, 1]
        assert np.all(np.logical_and(y_pred_pb >= 0, y_pred_pb <= 1))

    def test_fit(self):
        # Create a LogisticRegression instance
        lr = LogisticRegression(in_features=10)

        # Generate random data for testing
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model on the training data
        lr.fit(X_train, y_train)

        # Predict the output for the test data
        y_pred = lr.predict(X_test)

        # Check if the predicted output has the correct shape
        assert y_pred.shape == (X_test.shape[0],)

        # Check if the predicted output contains only 0s and 1s
        assert np.all(np.logical_or(y_pred == 0, y_pred == 1))

        # Check if the predicted output is within the range [0, 1]
        assert np.all(np.logical_and(y_pred >= 0, y_pred <= 1))

        # Check if the predicted output is the same as the predicted probability
        assert np.all(y_pred == (lr.predict_proba(X_test) > 0.5))

    def test_score(self):
        # Create a LogisticRegression instance
        lr = LogisticRegression(in_features=10)

        # Generate random data for testing
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model on the training data
        lr.fit(X_train, y_train, epochs=10, lr=0.01)

        # Calculate the accuracy score
        score = accuracy(lr.predict(X_test), y_test)

        # Check if the calculated score is within the range [0, 1]
        print(score)
        assert score >= 0.5 and score <= 1

# Run the tests
if __name__ == '__main__':
    pytest.main()