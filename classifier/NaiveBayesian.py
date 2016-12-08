#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/29 22:04
# @Author  : wqs
# @File    : NaiveBayesian.py

import sklearn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer


class NaiveBayesian(sklearn.base.BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None

    def fit(self, X, y):
        _, n = X.shape
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes = labelbin.classes_
        Y = np.concatenate((1 - Y, Y), axis=1).astype(np.float64)

        class_count = np.zeros(2, dtype=np.float64)
        feature_count = np.zeros((2, n), dtype=np.float64)

        feature_count += safe_sparse_dot(Y.T, X)  # count frequency by y.T * X
        class_count += Y.sum(axis=0)

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        self.feature_log_prob = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))
        # self.class_log_prior = np.zeros(2) - np.log(2)
        self.class_log_prior = np.log(class_count / sum(class_count))
        return self

    def predict(self, X):
        jll = safe_sparse_dot(X, self.feature_log_prob.T) + self.class_log_prior
        return self.classes[np.argmax(jll, axis=1)]
