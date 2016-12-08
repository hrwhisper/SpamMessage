# -*- coding: utf-8 -*-
# @Date    : 2016/10/16
# @Author  : hrwhisper
import random
import sklearn
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot


class LogisticRegression(sklearn.base.BaseEstimator):
    def __init__(self, alpha=0.1, max_iter=100):  # 0.02 200 0.9927425
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None

    def _sigmoid(self, x):
        return .5 * (1 + np.tanh(.5 * x))  # 1.0 / (1 + np.exp(-x))

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
            h = self._sigmoid(safe_sparse_dot(X, self.weights))  # X * weights
            error = target - h
            self.weights += self.alpha * safe_sparse_dot(X.T, error)  # alpha * X.T * error
        return self

    def predict(self, X):
        return (self._sigmoid(safe_sparse_dot(X, self.weights)) > 0.5).ravel().astype('int')


if __name__ == '__main__':
    a = [1, 2, 3]
    b = [4, 5, 6]
    a = np.array(a)
    b = np.array(b)
    print(a, b)
    a = a.reshape((-1, 1))
    print(a)
    print(a - b)
