import numpy as np
import pandas as pd
import unittest

from ppyPatternRecognition.knn import KNN

class KNNTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'label': [0, 0, 1, 1, 1]
        })

    def test_fit(self):
        knn = KNN()
        df = knn.fit(df=self.data, k=2, inplace=False, explain=False)
        self.assertEqual(df['label'].nunique(), 2)

    def test_mean_point(self):
        knn = KNN()
        mean = knn.mean_point(self.data)
        expected_mean = np.array([3.0, 6.0])
        np.testing.assert_array_equal(mean, expected_mean)

    def test_distance(self):
        knn = KNN()
        x1 = np.array([1, 2])
        x2 = np.array([4, 6])
        distance = knn.distance(x1, x2)
        expected_distance = 5.0
        self.assertAlmostEqual(distance, expected_distance)
    
    def test_explain(self):
        knn = KNN()
        df = knn.fit(df=self.data, k=2, inplace=False, explain=True)
        self.assertEqual(df['label'].nunique(), 2)
        

if __name__ == '__main__':
    unittest.main()