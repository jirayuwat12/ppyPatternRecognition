import numpy as np 

from ppyPatternRecognition.regression.linear_regression import LinearRegression
from ppyPatternRecognition.activation_function import sigmoid

class LogisticRegression(LinearRegression):
    def __init__(self,
                 in_features=None,
                 init_weights=None,
                 init_weights_method='zeros'):
        """
        Initialize the LogisticRegression class.

        Args:
            in_features (int): Number of input features. Defaults to None.
            init_weights (ndarray): Initial weights for the model. Defaults to None.
            init_weights_method (str): Method for initializing weights. Defaults to 'zeros'.
        """
        super().__init__(in_features, init_weights, init_weights_method)

    def predict(self,
                X,
                threshold=0.5):
        """
        Predicts the class labels for the input samples.

        Args:
            X (ndarray): Input samples.
            threshold (float): Threshold value for classification. Defaults to 0.5.

        Returns:
            ndarray: Predicted class labels.
        """
        X = np.array(X)
        return sigmoid(np.dot(X, self.weights.reshape(-1, 1)).reshape(-1)) > threshold
    
    def predict_proba(self,
                      X):
        """
        Predicts the class probabilities for the input samples.

        Args:
            X (ndarray): Input samples.

        Returns:
            ndarray: Predicted class probabilities.
        """
        X = np.array(X)
        return sigmoid(np.dot(X, self.weights.reshape(-1, 1)).reshape(-1))
