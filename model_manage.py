# -*- coding: utf-8 -*-
# @Date    : 2016/12/1
# @Author  : hrwhisper
from sklearn.externals import joblib


class BowTransform(object):
    default_path = './model/vsm.pkl'

    @staticmethod
    def save_vsm(model, filename=None):
        joblib.dump(model, filename if filename else BowTransform.default_path)

    @staticmethod
    def load_vsm(filename=None):
        return joblib.load(filename if filename else BowTransform.default_path)


class TrainData(object):
    default_path = './model/train_data.pkl'

    @staticmethod
    def save(model, filename=None):
        joblib.dump(model, filename if filename else TrainData.default_path)

    @staticmethod
    def load(filename=None):
        with open('./data/tags_token_results' + '_tag') as f:
            return joblib.load(filename if filename else TrainData.default_path), list(map(int, f.read().split('\n')[:-1]))
