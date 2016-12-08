# -*- coding: utf-8 -*-
# @Date    : 2016/12/3
# @Author  : hrwhisper
import codecs
import sys
from collections import Counter
from optparse import OptionParser
from multiprocessing import Pool
import jieba
from sklearn.externals import joblib
from model_manage import BowTransform
import time


def token(x):
    return Counter(jieba.lcut(x))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-c', '--classifier', dest="cls_name", type='string', default='p',
                      help="define the classifier you want to use:  \t\t\n"
                           "p  => Perceptron,\t\t\t\t\t\n"
                           "lr => LogisticRegression,\t\t\t\t\t\n"
                           "nb => NaiveBayesian,\t\t\t\t\t\n"
                           "svm => SVM(sklearn),\t\t\t\t\t\n"
                           "lrs => LogisticRegression(sklearn),\t\t\t\t\t\n"
                           "nbs => NaiveBayesian(sklearn),")

    parser.add_option('-i', '--input', dest="input_filename", type='string', default='./data/不带标签短信.txt',
                      help="input file name")

    parser.add_option('-o', '--output', dest="output_filename", type='string', default='./data/result.txt',
                      help="output file name")

    options, args = parser.parse_args()
    #
    classifiers = {
        'p': './model/Perceptron.pkl',  # 0.1 2000
        'lr': './model/LogisticRegression.pkl',  # 0.2 2000
        'nb': './model/NaiveBayesian.pkl',  # 0.00241
        'svm': './model/SVM_sklearn.pkl',
        'lrs': './model/Logistic_sklearn.pkl',
        'nbs': './model/Bayes_sklearn.pkl'
    }
    #
    cls_name = options.cls_name
    file_path = options.input_filename
    out_path = options.output_filename

    if cls_name not in classifiers.keys():
        print('check your classifiers name, you can use -h for help')
        sys.exit()

    start = time.time()
    jieba.initialize()
    try:
        with codecs.open(file_path, 'r', 'utf-8') as f:
            data = [line.strip() for line in f.read().split('\n')]
            if data[-1] == '':
                data.pop()
    except FileNotFoundError as e:
        print('Please check your input filename')
        sys.exit()

    # data = [Counter(d) for d in map(jieba.cut, data)]
    data = Pool().map(token, data)
    print('end token in {}\n'.format(time.time() - start))
    cv = BowTransform.load_vsm()
    data = cv.transform(data)
    print('end bow in {}\n'.format(time.time() - start))
    cls = joblib.load(classifiers[cls_name])
    predicted = cls.predict(data)

    # print(predicted)
    with open(out_path, 'w+') as f:
        for x in predicted:
            f.write(str(x) + '\n')
    print('task complete. total time: {}\n using {}'.format(time.time() - start, cls))
