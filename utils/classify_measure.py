# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/3

class ClassifyMeasure():
    """
    分类算法衡量指标
    """

    @staticmethod
    def accuracy_score(y_true, y_predict):
        """
        计算分类算法中y_true和y_predict的准确率
        :param y_true:
        :param y_predict:
        :return:
        """
        assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"
        return sum(y_true == y_predict) / len(y_true)
