import pytest
import numpy as np

import sys
sys.path.append('../ppyPatternRecognition')
from ppyPatternRecognition.loss_function import MSE_loss, L1_loss

def test_MSE_loss():
    # Test case 1: Same predicted and actual values
    y_pred = np.array([1, 2, 3])
    y_actual = np.array([1, 2, 3])
    assert MSE_loss(y_pred, y_actual) == 0

    # Test case 2: Different predicted and actual values
    y_pred = np.array([1, 2, 3])
    y_actual = np.array([4, 5, 6])
    assert MSE_loss(y_pred, y_actual) == 9

    # Test case 3: Empty predicted and actual values
    y_pred = np.array([])
    y_actual = np.array([])
    assert MSE_loss(y_pred, y_actual) == 0

    # Test case 4: Single predicted and actual value
    y_pred = np.array([5])
    y_actual = np.array([10])
    assert MSE_loss(y_pred, y_actual) == 25

    # Test case 5: Large predicted and actual values
    y_pred = np.array([1000, 2000, 3000])
    y_actual = np.array([500, 1000, 1500])
    assert MSE_loss(y_pred, y_actual) >=0

    # Test case 6: Negative predicted and actual values
    y_pred = np.array([-1, -2, -3])
    y_actual = np.array([-4, -5, -6])
    assert MSE_loss(y_pred, y_actual) == 9

    # Test case 8: Large number of predicted and actual values
    y_pred = np.random.rand(1000)
    y_actual = np.random.rand(1000)
    assert MSE_loss(y_pred, y_actual) >= 0

    # Test case 9: NaN values in predicted and actual values
    y_pred = np.array([1, 2, np.nan])
    y_actual = np.array([1, 2, 3])
    assert np.isnan(MSE_loss(y_pred, y_actual))

    # Test case 10: Infinite values in predicted and actual values
    y_pred = np.array([1, 2, np.inf])
    y_actual = np.array([1, 2, 3])
    assert np.isinf(MSE_loss(y_pred, y_actual))


def test_L1_loss_empty_arrays():
    # Empty arrays
    y_pred = np.array([])
    y_actual = np.array([])

    # Calculate L1 loss
    loss = L1_loss(y_pred, y_actual)

    # Check if loss is 0
    assert loss == 0


def test_L1_loss_same_lengths():
    # Arrays with same lengths
    y_pred = np.array([1, 2, 3])
    y_actual = np.array([4, 5, 6])

    # Calculate L1 loss
    loss = L1_loss(y_pred, y_actual)

    # Check if loss is correct
    assert loss == 3.0

def test_L1_loss_negative_values():
    # Arrays with negative values
    y_pred = np.array([-1, -2, -3])
    y_actual = np.array([4, 5, 6])

    # Calculate L1 loss
    loss = L1_loss(y_pred, y_actual)

    # Check if loss is correct
    assert loss == 7.0