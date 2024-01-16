class Regression():
    """
    Base class for regression models.

    Methods:
    - fit(X, y): Fit the regression model to the training data.
    - predict(X): Predict the target variable for new input data.
    - score(X, y): Evaluate the performance of the regression model.
    - get_params(deep=True): Get the parameters of the regression model.
    - set_params(**params): Set the parameters of the regression model.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass