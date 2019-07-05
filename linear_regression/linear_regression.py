# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/3
import numpy as np
from utils.regression_measure import RegressionMeasure
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        """
        初始化Linear Regression 模型
        """
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """
        训练函数
        :param X_train: 训练数据集输入
        :param y_train: 训练数据集输出
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, learning_rate=0.01, n_iters=1e4):
        """
        使用梯度下降的方法来训练Linear Regression模型
        :param X_train: 训练数据集特征
        :param y_train: 训练数据集的结果
        :param learning_rate: 学习率
        :param iter: 最大迭代次数
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """
            计算损失函数
            :param theta:
            :param X_b:
            :param y:
            :return:
            """
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            """
            计算梯度
            :param theta: 参数
            :param X_b: 训练数据集
            :param y:
            :return:
            """
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, learning_rate, n_iters=1e4, epsilon=1e-8):
            """
            进行梯度下降
            :param X_b:
            :param initial_theta:
            :param learning_rate:
            :param n_iters:
            :param epsilon:
            :return:
            """
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - learning_rate * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, learning_rate, n_iters)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """
        预测函数
        :param X_predict: 待预测数据集合
        :return: 预测结果的集合
        """

        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """
        模型评价函数
        :param X_test: 测试数据集的输入
        :param y_test: 测试数据集的输出
        :return: R squared的值
        """
        y_predict = self.predict(X_test)
        return RegressionMeasure.r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"


if __name__ == '__main__':
    regression = LinearRegression()

    np.random.seed(666)
    x = 2 * np.random.random(size=100)
    y = x * 3 + 4. + np.random.normal(size=100)
    X = x.reshape(-1, 1)
    plt.scatter(x, y)
    plt.show()

    regression.fit_gd(X, y)
    print(regression.coef_)
    print(regression.interception_)
