# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/3

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

    @staticmethod
    def confusion_matrix(y_true, y_predict):
        """

        :param y_true:
        :param y_predict:
        :return:
        """

        def TN(y_true, y_predict):
            """
            计算混淆矩阵中的True Negative
            :param y_true:
            :param y_predict:
            :return:
            """
            assert len(y_true) == len(y_predict)
            return np.sum((y_true == 0) & (y_predict == 0))  # 里面的结果是一个布尔向量

        def FP(y_true, y_predict):
            """
            计算混淆矩阵中的False positive
            :param y_true:
            :param y_predict:
            :return:
            """
            assert len(y_true) == len(y_predict)
            return np.sum((y_true == 0) & (y_predict == 1))  # 里面的结果是一个布尔向量

        def FN(y_true, y_predict):
            """
            计算混淆矩阵中的True Negative
            :param y_true:
            :param y_predict:
            :return:
            """
            assert len(y_true) == len(y_predict)
            return np.sum((y_true == 1) & (y_predict == 0))  # 里面的结果是一个布尔向量

        def TP(y_true, y_predict):
            """
            计算混淆矩阵中的True Negative
            :param y_true:
            :param y_predict:
            :return:
            """
            assert len(y_true) == len(y_predict)
            return np.sum((y_true == 1) & (y_predict == 1))  # 里面的结果是一个布尔向量

        return {
            "TN": TN(y_true, y_predict),
            "FP": FP(y_true, y_predict),
            "FN": FN(y_true, y_predict),
            "TP": TP(y_true, y_predict)
        }

    @staticmethod
    def precision_score(y_true, y_predict):
        """
        计算分类模型准确度
        :param y_true: 真实类别
        :param y_predict: 预测类别
        :return:
        """
        tp = ClassifyMeasure.confusion_matrix(y_true, y_predict).get("TP")
        fp = ClassifyMeasure.confusion_matrix(y_true, y_predict).get("FP")
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def recall_score(y_true, y_predict):
        """
        计算分类模型召回率
        :param y_true: 真实类别
        :param y_predict: 预测类别
        :return:
        """
        tp = ClassifyMeasure.confusion_matrix(y_true, y_predict).get("TP")
        fn = ClassifyMeasure.confusion_matrix(y_true, y_predict).get("FN")
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def f1_score(precision, recall):
        """
        根据准确率和召回率计算f1值
        :param precision: 精确率
        :param recall: 召回率
        :return:
        """
        try:
            return 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return 0.0

if __name__ == '__main__':
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target.copy()

    # 将一个多分类问题转为一个二分类问题，手动产生较大偏斜的数据集
    y[digits.target == 9] = 1
    y[digits.target != 9] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)  # 对数据集进行划分

    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train, y_train)
    print(ClassifyMeasure.confusion_matrix(y_test, log_reg.predict(X_test)))
    print(ClassifyMeasure.precision_score(y_test, log_reg.predict(X_test)))
    print(ClassifyMeasure.recall_score(y_test, log_reg.predict(X_test)))
