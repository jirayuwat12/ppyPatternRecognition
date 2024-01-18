import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters:
    x (float): The input value.

    Returns:
    float: The output value after applying the sigmoid function.
    """
    return 1/(1+np.exp(-x))