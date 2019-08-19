# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/5
import numpy as np
import matplotlib.pyplot as plt
from utils.classify_measure import ClassifyMeasure
from sklearn import datasets
from utils import model_selection
from utils.decision_boundary import DecisionBoundary


class LogisticRegression:
    """
    逻辑回归模型
    """

    def __init__(self):
        """
        初始化Linear Regression 模型
        """
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit(self, X_train, y_train, learning_rate=0.01, n_iters=1e4):
        """
        使用批量梯度下降的方法来训练Linear Regression模型
        :param X_train: 训练数据集特征
        :param y_train: 训练数据集的结果
        :param learning_rate: 学习率
        :param iter: 最大迭代次数
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """
            逻辑回归算法中的损失函数
            :param theta:
            :param X_b:
            :param y:
            :return:
            """
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
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
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

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

    def fit_1(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        """
        对于待预测数据集X_predict,返回预测的结果概率向量
        :param X_predict: 待预测数据集合
        :return: 预测结果的集合
        """

        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """
        预测函数
        :param X_predict: 待预测数据集合
        :return: 预测结果的集合
        """

        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """

        :param X_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(X_test)
        return ClassifyMeasure.accuracy_score(y_test, y_predict)

    @staticmethod
    def _sigmoid(t):
        """
        sigmoid函数
        :param t:
        :return:
        """
        return 1 / (1 + np.exp(-t))

    def __repr__(self):
        return "Logistic Regression"


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y < 2, :2]  # 做二分类，鸢尾花有三种数据，因此取前两类，要做可视化，因此取前两个特征
    y = y[y < 2]

    # 使用逻辑回归进行分类
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, seed=666)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)

    # 决策边界的两个参数

    x1_plot = np.linspace(4, 8, 1000)
    x2_plot = (-logistic_regression.coef_[0] * x1_plot - logistic_regression.interception_) / logistic_regression.coef_[
        1]

    # 数据及决策边界的可视化
    DecisionBoundary.plot_decision_boundary(logistic_regression, axis=[4, 7.5, 1.5, 4.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
    plt.show()
