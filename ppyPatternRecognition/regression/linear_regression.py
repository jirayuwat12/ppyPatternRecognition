import numpy as np

from .base_regression import Regression

class LinearRegression(Regression):
    """
    Linear regression model.

    Parameters:
    - init_weights: Initial weights for the linear regression model. If None, the weights will be initialized as None.

    Methods:
    - fit(X, y): Fit the linear regression model to the training data.
    - predict(X): Predict the target variable for the given input data.
    - score(X, y): Calculate the coefficient of determination (R^2) of the linear regression model.
    - get_params(deep=True): Get the parameters of the linear regression model.
    - set_params(**params): Set the parameters of the linear regression model.
    """
    def __init__(self,
                 init_weights=None):
        if init_weights is not None:
            if not isinstance(init_weights, np.ndarray):
                self.weights = np.array(init_weights)
            else:
                self.weights = init_weights
        else:
            self.weights = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
