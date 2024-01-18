import numpy as np

def L1_loss(y_pred, y_actual):
    """
    Calculate the L1 loss between the predicted values and the actual values.

    Parameters:
    y_pred (numpy.ndarray): The predicted values.
    y_actual (numpy.ndarray): The actual values.

    Returns:
    float: The mean absolute error (MAE) if the lengths of y_pred and y_actual are greater than 0 and equal, otherwise 0.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)

    return np.abs(y_pred - y_actual).mean()


def MSE_loss(y_pred, y_actual):
    """
    Calculates the mean squared error (MSE) loss between predicted and actual values.

    Parameters:
    - y_pred (array-like): Predicted values.
    - y_actual (array-like): Actual values.

    Returns:
    - mse (float): Mean squared error loss.

    Note:
    - The lengths of y_pred and y_actual must be equal.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)

    return ((y_pred - y_actual)**2).mean()


def L2_loss(y_pred, y_actual):
    """
    Calculates the L2 loss between the predicted values and the actual values.

    Parameters:
    y_pred (array-like): Predicted values.
    y_actual (array-like): Actual values.

    Returns:
    float: L2 loss between the predicted values and the actual values.
    """
    return MSE_loss(y_pred, y_actual)