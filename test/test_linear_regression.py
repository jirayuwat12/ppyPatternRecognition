import pytest
import numpy as np
from sklearn.datasets import make_regression

import sys
sys.path.append('../ppyPatternRecognition')
from ppyPatternRecognition.regression.linear_regression import LinearRegression

def test_predict():
    lr = LinearRegression(in_features=1,
                          init_weights=[5])

    assert np.allclose(lr.predict([[1], [3], [5]]), [5, 15, 25])

def test_fit():
    lr = LinearRegression(in_features=2,
                          init_weights_method='zeros')
    x = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [5, 8, 11, 14]
    lr.fit(X=x,
           y=y,
           epochs=100)
    
    assert np.allclose([round(i) for i in lr.predict(x)[0]], y)
