import numpy as np

from ppyPatternRecognition.regression.base_regression import Regression
from ppyPatternRecognition.loss_function import MSE_loss


class LinearRegression(Regression):
    def __init__(self,
                 in_features = None,
                 init_weights=None,
                 init_weights_method='zeros'):
        """
        Linear regression model.

        Parameters:
        - in_features (int): Number of input features.
        - init_weights (list or None): Initial weights for the linear regression model. If None, the weights will be initialized using the specified method.
        - init_weights_method (str): Method for initializing the weights. Options are 'zeros' and 'random'.

        Raises:
        - ValueError: If in_features is None.

        """
        if in_features is None:
            raise ValueError
        self.in_features = in_features

        if init_weights is not None and len(init_weights) == in_features:
            self.weights = np.array(init_weights)
        elif init_weights_method in ['zeros', 'random']:
            if init_weights_method == 'zeros':
                self.weights = np.zeros((in_features, 1))
            elif init_weights_method == 'random':
                self.weights = np.random.random((in_features, 1))


    def fit(self, X, y, epochs=1, lr=1e-3):
        """
        Fit the linear regression model to the training data.

        Parameters:
        - X (array-like): Training input data.
        - y (array-like): Target values.
        - epochs (int): Number of training epochs.
        - lr (float): Learning rate.

        """
        X = np.array(X)
        y = np.array(y)

        for epoch in range(epochs):
            # calculate gradient
            y_pred = self.predict(X).reshape(1, -1)
            gradient = ((y-y_pred.reshape(1, -1)).reshape(-1, 1)*X).sum(axis=0)
            # update weight
            self.weights += (lr*gradient).reshape(-1, 1)
            print(f"Epoch : {epoch+1}/{epochs} | Loss : {MSE_loss(y_pred, y)}")

    def predict(self, X):
        """
        Predict the target values for the input data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - array-like: Predicted target values.

        """
        X = np.array(X)
        return np.dot(X, self.weights.reshape(-1, 1)).reshape(1, -1)


    def get_params(self, deep=True):
        """
        Get the parameters of the linear regression model.

        Parameters:
        - deep (bool): Whether to return a deep copy of the parameters.

        Returns:
        - dict: Parameters of the linear regression model.

        """
        return {
            'weights' : self.weights,
            'in_features' : self.in_features
        }

    def set_params(self, **params):
        """
        Set the parameters of the linear regression model.

        Parameters:
        - params (dict): Parameters to be set.

        Returns:
        - self: The linear regression model instance.

        """
        for key, value in params.items():
            setattr(self, key, value)
        return self