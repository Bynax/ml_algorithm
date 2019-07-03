# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/3
import numpy as np
from math import sqrt


class RegressionMeasure():
    """
    简单线性回归的衡量函数
    """

    @staticmethod
    def mean_squared_error(y_true, y_predict):
        """
        计算y_true与y_predict之间的MSE
        :param y_true: 真实数据
        :param y_predict: 预测数据
        :return:
        """
        assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
        return np.sum((y_true - y_predict) ** 2) / len(y_true)

    @staticmethod
    def root_mean_squared_error(y_true, y_predict):
        """
        计算y_true与y_predict之间的RMSE
        :param y_true: 真实数据
        :param y_predict: 预测数据
        :return:
        """
        return sqrt(RegressionMeasure.mean_squared_error(y_true, y_predict))

    @staticmethod
    def mean_absolute_error(y_true, y_predict):
        """
        计算y_true与y_predict之间的MAE
        :param y_true: 真实数据
        :param y_predict: 预测数据
        :return:
        """
        assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
        return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

    @staticmethod
    def r2_score(y_true, y_predict):
        """
        计算y_true与y_predict之间的R Square
        :param y_true:
        :param y_predict:
        :return:
        """
        return 1 - RegressionMeasure.mean_squared_error(y_true, y_predict) / np.var(y_true)
