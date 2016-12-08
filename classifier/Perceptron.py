# -*- coding: utf-8 -*-
# @Date    : 2016/11/1
# @Author  : hrwhisper

import numpy as np
import sklearn
from sklearn.utils.extmath import safe_sparse_dot


class Perceptron(sklearn.base.BaseEstimator):
    def __init__(self, alpha=0.1, max_iter=100):
        self.threshold = 0.5
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None

    def fit(self, X, y):
        """
            :param X: sparse matrix(n_samples,n_features) the training feature vector
            :param y:  array-like(n_samples,) Target vector relative to X.
            :return: self
        """
        m, n = X.shape
        target = np.array(y).reshape((m, 1))
        self.weights = np.ones((n, 1))
        for _ in range(self.max_iter):
            h = safe_sparse_dot(X, self.weights) > self.threshold
            error = target - h
            self.weights += self.alpha * safe_sparse_dot(X.T, error)
        return self

    def predict(self, X):
        return (safe_sparse_dot(X, self.weights) > self.threshold).ravel().astype('int')
