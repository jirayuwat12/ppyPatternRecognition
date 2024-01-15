from enum import unique
import pytest

from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

import sys
sys.path.append('../ppyPatternRecognition')
from ppyPatternRecognition import KNN

def test_knn_stop_before_max():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=False, max_iter=10)
    assert df is not None

def test_knn_stop_at_max():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=False, max_iter=1)
    assert df is not None

def test_knn_stop_at_max_explain():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=True, max_iter=1)
    assert df is not None

def test_knn_stop_before_max_explain():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=True, max_iter=10)
    assert df is not None

def test_knn_correctness():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=False, max_iter=100)
    unique_label = df['label'].unique()
    assert len(unique_label) == 5

def test_knn_correctness_explain():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=True, max_iter=100)
    unique_label = df['label'].unique()
    assert len(unique_label) == 5

def test_knn_correctness_many_features():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=0)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, 11)])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=False, max_iter=100)
    unique_label = df['label'].unique()
    assert len(unique_label) == 5

def test_knn_correctness_many_features_explain():
    # generate data
    X, y = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=0)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, 11)])
    df['label'] = y

    # KNN
    knn = KNN()
    df = knn.fit(df, k=5, explain=True, max_iter=100)
    unique_label = df['label'].unique()
    assert len(unique_label) == 5
