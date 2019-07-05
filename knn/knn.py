# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/6/24
import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    """
    构建简单的KNN分类器
    """

    def __init__(self, k):
        """
        初始化方法
        :param k: 以最近k个为标准
        """
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None  # 约定二维或以上的向量使用大写
        self._y_train = None  # 一维的使用小写

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            " the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        对外开放的预测方法
        :param X_predict: 要预测的样本
        :return: 表示样本预测的结果向量
        """
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        # 对X_predict的每一个元素进行预测
        y_predict = [self._predict(x) for x in X_predict]
        # 返回预测结果
        return np.array(y_predict)

    def _predict(self, x):
        """
        对单个样本进行预测
        :param x: 单个样本向量
        :return: 预测结果
        """
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must equal to X_train"
        # 计算x与_X_train中每一个点的距离
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        # np.argsort方法 将索引按照distances进行排序
        nearest = np.argsort(distances)
        # 根据x的索引找到对应的_y_train
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        # 使用Counter封装
        votes = Counter(topK_y)

        # 返回最频繁的label
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=)".format(self.k)
