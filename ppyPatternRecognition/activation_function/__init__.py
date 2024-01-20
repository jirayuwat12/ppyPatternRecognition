import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters:
    - x (float, int, or ndarray): Input value.

    Returns:
    float: The output value after applying the sigmoid function.
    """
    if isinstance(x, np.ndarray):
        x[x>100] = 100.0
        x[x<-100] = -100.0
    else:
        if x > 100:
            x = 100.0
        elif x < -100:
            x = -100.0

    return 1/(1+np.exp(-x))
