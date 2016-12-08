# -*- coding: utf-8 -*-
# @Date    : 2016/12/3
# @Author  : hrwhisper
import codecs
import sys
from collections import Counter
from multiprocessing import Pool
from optparse import OptionParser
import jieba
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from model_manage import BowTransform
from test_jieba import test_not_tag_data


def test_data():
    def read_train_data():
        file_path = './data/带标签短信.txt'
        target = []
        data = []
        with codecs.open(file_path, 'r', 'utf-8') as f:
            for line in f.read().split('\n')[:-1]:
                line = line.strip()
                target.append(line[0])
                data.append(line[1:].lstrip())
        return data, target

    data, target = read_train_data()
    n = len(data) - int(0.8 * len(data))
    return data[-n:], list(map(int, target[-n:]))


def f(x):
    return Counter(jieba.cut(x))


if __name__ == "__main__":
    classifiers = {
        'p': './model/Perceptron.pkl',  # 0.1 2000
        'lr': './model/LogisticRegression.pkl',  # 0.2 2000
        'nb': './model/NaiveBayesian.pkl',  # 0.00241
        'svm': './model/SVM_sklearn.pkl',
        'lrs': './model/Logistic_sklearn.pkl',
        'nbs': './model/Bayes_sklearn.pkl'
    }
    import time

    # for cls_name in classifiers.keys():
    #     jieba.initialize()
    #
    #     start = time.time()
    #     data = test_not_tag_data()
    #     cls = joblib.load(classifiers[cls_name])
    #
    #     data = Pool().map(f, data)
    #     # data = [Counter(d) for d in map(jieba.cut, data)]
    #     print('end jieba', time.time() - start)
    #     cv = BowTransform.load_vsm()
    #     data = cv.transform(data)
    #     predicted = cls.predict(data)
    #     with open('./data/result.txt', 'w+') as f:
    #         for x in predicted:
    #             f.write(str(x) + '\n')
    #     print('end %s  with time:' % cls, time.time() - start)

    start = time.time()
    data, target = test_data()
    # data = [Counter(d) for d in map(jieba.cut, data)]
    data = Pool(4).map(f, data)
    cv = BowTransform.load_vsm()
    data = cv.transform(data)

    for cls_name in classifiers.keys():
        cls = joblib.load(classifiers[cls_name])
        predicted = cls.predict(data)

        print(predicted.shape)  # 160 1

        # print(sum(predicted == test_target), len(test_target), np.mean(predicted == test_target))
        print("Classification report for classifier %s:\n%s\n" % (
            cls, metrics.classification_report(target, predicted, digits=4)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(target, predicted))
        print(precision_recall_fscore_support(target, predicted))
        print('end', time.time() - start)
