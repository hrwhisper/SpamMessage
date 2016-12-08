# -*- coding: utf-8 -*-
# @Date    : 2016/10/16
# @Author  : hrwhisper
import codecs
from collections import Counter
import datetime

from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import metrics, naive_bayes, svm, linear_model
from classifier.LogisticRegression import LogisticRegression
from classifier.NaiveBayesian import NaiveBayesian
from classifier.Perceptron import Perceptron
from model_manage import BowTransform, TrainData


def read_train_data():
    file_path = './data/tags_token_results'
    with codecs.open(file_path, 'r', 'utf-8') as f:
        data = [line.strip().split() for line in f.read().split('\n')]

    with open(file_path + '_tag') as f:
        return data[:-1], list(map(int, f.read().split('\n')[:-1]))


def _test(classifier, test_data, test_target):
    predicted = classifier.predict(test_data)
    print(predicted.shape)  # 160 1

    # print(sum(predicted == test_target), len(test_target), np.mean(predicted == test_target))
    print("Classification report for classifier %s:\n%s\n" % (
        classifier, metrics.classification_report(test_target, predicted, digits=4)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_target, predicted))
    print(precision_recall_fscore_support(test_target, predicted))


def test_one(cls, use_save_data=True, train_cls=False, save_cls_path=None):
    if use_save_data:
        data, target = TrainData.load()
    else:
        data, target = read_train_data()
        # data_len = int(len(data) * 0.001)
        # data, target = data[:data_len], target[:data_len]

        data = [Counter(d) for d in data]  # 每一行为一个短信， 值就是TF
        # print(data[0])
        v = DictVectorizer()
        print('fit transform')
        data = v.fit_transform(data)  # 稀疏矩阵表示sparse matrix,词编好号
        TrainData.save(data)

    # print(data[0])
    data_len = data.shape[0]
    print('data', data.shape[1])
    end = int(0.8 * data_len)
    train_data, train_target = data[:end], target[:end]
    test_data, test_target = data[end:], target[end:]

    if train_cls:
        print('train classifier....')
        cls = cls.fit(train_data, train_target)
        print('train classifier complete')

    _test(cls, test_data, test_target)

    if save_cls_path:
        joblib.dump(cls, save_cls_path)


def cross_validation():
    data, target = TrainData.load()

    classifiers = {
        'Logistic by yhr': LogisticRegression(alpha=0.2, max_iter=2000),
        'Perceptron by yhr': Perceptron(alpha=0.1, max_iter=2000),
        'Bayesian by wqs': NaiveBayesian(),
        'Bernoulli Bayes from sklearn': naive_bayes.BernoulliNB(),
        'svm from sklearn': svm.LinearSVC(),
        'Logistic from sklearn': linear_model.LogisticRegression(),
        # 'decision tree':tree.DecisionTreeClassifier(),
    }

    for name, classifier in classifiers.items():
        this_scores = cross_val_score(classifier, data, target, cv=5, scoring='accuracy')
        print(name)
        print(this_scores)
        print(np.mean(this_scores))
        print(' ------------------------  \n\n')


def test_parameter():
    data, target = TrainData.load()
    max_score = 0
    max_alpha = max_iter = 0
    print('Perceptron')
    start = datetime.datetime.now()
    for alpha in [0.01, 0.1, 0.2]:  # 0.3 100 0.99133
        for iter in [100, 2000]:
            cls = Perceptron(alpha=alpha, max_iter=iter)
            this_scores = cross_val_score(cls, data, target, cv=5, scoring='accuracy')
            print(this_scores)
            cur = np.mean(this_scores)
            print(alpha, iter, cur)
            print(' ------------------------  \n\n')
            if cur > max_score:
                max_score = cur
                max_alpha, max_iter = alpha, iter
            print('current_max: ', max_score, max_alpha, max_iter)
    print((datetime.datetime.now() - start))


if __name__ == '__main__':
    # start = datetime.datetime.now()
    # test_one(LogisticRegression(alpha=0.1, max_iter=2000),
    #          train_cls=True, save_cls_path='./model/LogisticRegression.pkl')
    # print((datetime.datetime.now() - start))
    #
    # classifiers = {
    #     # 'Logistic by yhr': LogisticRegression(alpha=0.01, max_iter=200),
    #     # 'Perceptron by yhr': Perceptron(),
    #     # 'Bayesian by wqs': NaiveBayesian(),
    #     # 'Bayes_sklearn': naive_bayes.BernoulliNB(),
    #     # 'SVM_sklearn': svm.LinearSVC(),
    #     # 'Logistic_sklearn': linear_model.LogisticRegression(),
    #     # 'decision tree':tree.DecisionTreeClassifier(),
    # }
    # for name, cls in classifiers.items():
    #     test_one(cls, train_cls=True, save_cls_path='./model/' + name + '.pkl')


    #
    # test_one(LogisticRegression(alpha=0.2, max_iter=2000), train_cls=True,
    #          save_cls_path='./model/LogisticRegression.pkl')
    # test_one(Perceptron(alpha=0.1, max_iter=2000), train_cls=True, save_cls_path='./model/Perceptron.pkl')
    #
    classifiers = {
        'p': './model/Perceptron.pkl',  # 0.1 2000
        'lr': './model/LogisticRegression.pkl',  # 0.2 2000
        'nb': './model/NaiveBayesian.pkl',  # 0.00241
        'svm': './model/SVM_sklearn.pkl',
        'lrs': './model/Logistic_sklearn.pkl',
        'nbs': './model/Bayes_sklearn.pkl'
    }

    for _path in classifiers.values():
        cls = joblib.load(_path)
        test_one(cls)

        # cross_validation()
        # test_one(LogisticRegression(max_iter=100), train_cls=True)
        # test_parameter()
