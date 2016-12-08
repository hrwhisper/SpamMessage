# -*- coding: utf-8 -*-
# @Date    : 2016/10/7
# @Author  : hrwhisper
from collections import Counter
from multiprocessing.pool import Pool
import jieba
import codecs

from sklearn.feature_extraction import DictVectorizer

from model_manage import TrainData


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


def save_tokenlization_result(data, target, file_path='./data/tags_token_results'):
    with codecs.open(file_path, 'w', 'utf-8') as f:
        for x in data:
            f.write(' '.join(x) + '\n')

    with open(file_path + '_tag', 'w') as f:
        for x in target:
            f.write(x + '\n')


if __name__ == '__main__':
    # seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    # print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    data, target = read_train_data()
    data = Pool().map(jieba.lcut, data)
    # data = jieba.lcut(data)
    save_tokenlization_result(data, target)

    with codecs.open('./data/tags_token_results', 'r', 'utf-8') as f:
        data = [line.strip().split() for line in f.read().split('\n')]
        if not data[-1]: data.pop()
        t = [Counter(d) for d in data]  # 每一行为一个短信， 值就是TF
        v = DictVectorizer()
        t = v.fit_transform(t)  # 稀疏矩阵表示sparse matrix,词编好号
        TrainData.save(t)
