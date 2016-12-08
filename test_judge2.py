# -*- coding: utf-8 -*-
# @Date    : 2016/12/3
# @Author  : hrwhisper
import codecs
import os
import sys
from collections import Counter
from multiprocessing import Pool
from optparse import OptionParser
import jieba
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from model_manage import BowTransform


def test_not_tag_data():
    with codecs.open('./data/不带标签短信.txt', 'r', 'utf-8') as f:
        data = f.read()
        return data
        #     if data[-1] == '':
        #         data.pop()
        # return data


def f(x):
    return Counter(jieba.cut(x))


if __name__ == "__main__":
    #
    classifiers = {
        'p': './model/Perceptron.pkl',  # 0.1 2000
        'lr': './model/LogisticRegression.pkl',  # 0.2 2000
        'nb': './model/NaiveBayesian.pkl',  # 0.00241
        'svm': './model/SVM_sklearn.pkl',
        'lrs': './model/Logistic_sklearn.pkl',
        'nbs': './model/Bayes_sklearn.pkl'
    }
    import time

    if os.name != 'nt':
        print('on linux enable parallel tantalization')
        jieba.enable_parallel(4)

    for cls_name in classifiers.keys():
        start = time.time()
        data = test_not_tag_data()
        cls = joblib.load(classifiers[cls_name])
        # data = Pool(4).map(f, data)

        data = [Counter(x) for x in ' '.join(jieba.cut(data)).split('\n')]
        # with codecs.open('./data/result.txt', 'w+', 'utf-8')  as f:
        #     f.write()

        print('end jieba', time.time() - start)
        cv = BowTransform.load_vsm()
        data = cv.transform(data)
        predicted = cls.predict(data)
        with open('./data/result.txt', 'w+') as f:
            for x in predicted:
                f.write(str(x) + '\n')
        print('end %s  with time:' % cls, time.time() - start)


        # data, target = test_data()
        # # data = [Counter(d) for d in map(jieba.cut, data)]
        # data = Pool(4).map(f, data)
        # cv = BowTransform.load_vsm()
        # data = cv.transform(data)


        # for cls_name in classifiers.keys():
        #     cls = joblib.load(classifiers[cls_name])
        #     predicted = cls.predict(data)
        #
        #     print(predicted.shape)  # 160 1
        #
        #     # print(sum(predicted == test_target), len(test_target), np.mean(predicted == test_target))
        #     print("Classification report for classifier %s:\n%s\n" % (
        #         cls, metrics.classification_report(target, predicted, digits=4)))
        #     print("Confusion matrix:\n%s" % metrics.confusion_matrix(target, predicted))
        #     print(precision_recall_fscore_support(target, predicted))
        #     print('end', time.time() - start)
