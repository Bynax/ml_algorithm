# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from measure.regression_measure import RegressionMeasure


class SimpleLinearRegression:
    """
    简单的一元线性回归
    """

    def __init__(self):
        """
        初始化
        """
        self.a_ = None  # 前面的横线表示私有变量，后面的横线表示不是用户送进来的变量，是自己根据用户输入计算的
        self.b_ = None

    def fit(self, x_train, y_train):
        """
        根据训练数据集训练SimpleLinearRegression模型
        :param x_train: 输入（只支持一维）
        :param y_train: 输出
        :return:
        """
        assert x_train.ndim == 1, "simple linear regression can only solve single feature training data."
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0  # 表示分子
        d = 0.0  # 表示分母

        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def fit_vector(self, x_train, y_train):
        """
        使用向量化的方式根据训练集训练SimpleLinearRegression模型
        :param x_train:输入
        :param y_train:输出
        :return:
        """
        assert x_train.ndim == 1, "simple linear regression can only solve single feature training data."
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """
        给定输入预测数据集，返回表示结果的向量
        :param x_predict: 要预测的列表
        :return:
        """
        assert x_predict.ndim == 1, "simple linear regression can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """
        对单个输入进行预测
        :param x_single:
        :return:
        """
        return self.a_ * x_single + self.b_

    def socre(self, x_test, y_test):
        """
        根据测试数据集x_test和y_test确定当前模型的准确度
        :param x_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(x_test)
        return RegressionMeasure.r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"


if __name__ == '__main__':
    regression = SimpleLinearRegression()
    boston = datasets.load_boston()
    x = boston.data[:, 5]
    y = boston.target
    x = x[y < 50.0]
    y = y[y < 50.0]

    regression.fit_vector(x, y)

    plt.scatter(x, y)
    plt.plot(x, regression.predict(x), color='r')
    plt.show()
