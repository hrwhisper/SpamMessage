# -*- coding: utf-8 -*-
# @Date    : 2016/12/5
# @Author  : hrwhisper
import codecs
import datetime
from collections import Counter
import jieba
from multiprocessing import Pool


def test_not_tag_data():
    with codecs.open('./data/不带标签短信.txt', 'r', 'utf-8') as f:
        data = [line.strip() for line in f.read().split('\n')]
        if data[-1] == '':
            data.pop()

    return data


def f(x):
    return Counter(jieba.cut(x))


if __name__ == "__main__":
    # jieba.enable_parallel(2)
    start = datetime.datetime.now()
    data = test_not_tag_data()
    print('read data', datetime.datetime.now() - start)
    # data = [Counter(d) for d in map(jieba.cut, data)]
    res = Pool(4).map(f, data)
    # print(res)
    print('jieba ', datetime.datetime.now() - start)

    # cv = BowTransform.load_vsm()
    # data = cv.transform(data)
    # print('transform', datetime.datetime.now() - start)
    # cls = joblib.load(classifiers[cls_name])
    # print('load', datetime.datetime.now() - start)
    # predicted = cls.predict(data)
    #
    # print(predicted)
    # with open('./data/result.txt', 'w+') as f:
    #     for x in predicted:
    #         f.write(str(x) + '\n')

    # print(datetime.datetime.now() - start, ' %s' % cls)
