import numpy as np 

from ppyPatternRecognition.regression.linear_regression import LinearRegression
from ppyPatternRecognition.activation_function import sigmoid
from ppyPatternRecognition.loss_function import MSE_loss
from ppyPatternRecognition.evaluation import accuracy


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

    def fit(self,
            X,
            y,
            epochs=1,
            lr=1e-3):
        """
        Fit the model to the training data.

        Args:
            X (ndarray): Training input data.
            y (ndarray): Target values.
            epochs (int): Number of training epochs. Defaults to 1.
            lr (float): Learning rate. Defaults to 1e-3.
        """
        X = np.array(X)
        y = np.array(y)

        for epoch in range(epochs):
            # calculate gradient
            y_pred = self.predict(X)
            gradient = ((y-y_pred.reshape(1, -1)).reshape(-1, 1)*X).sum(axis=0)
            # update weight
            self.weights += (lr*gradient).reshape(-1, 1)
            print(f"Epoch : {epoch+1}/{epochs} | Loss : {MSE_loss(y_pred, y)} | Accuracy : {accuracy(y_pred, y)}")
    

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
        return (self.predict_proba(X) >= threshold).astype(int)
    
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
