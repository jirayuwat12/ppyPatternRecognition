import numpy as np


def MSE_loss(y_pred, y_actual):
    """
    Calculate the MSE loss between predicted values and actual values.

    Parameters:
    y_pred (array-like): Predicted values.
    y_actual (array-like): Actual values.

    Returns:
    float: Mean squared error (MSE) between predicted and actual values.
    """

    e = y_pred - y_actual
    se = e**2
    mse = se.mean()

    return mse if len(y_pred) > 0 and len(y_pred) == len(y_actual) else 0
