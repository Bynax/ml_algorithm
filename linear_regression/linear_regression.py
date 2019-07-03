# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/3
import numpy as np
from utils.regression_measure import RegressionMeasure
from utils.model_selection import train_test_split
from sklearn import datasets


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

        :param X_train:
        :param y_train:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        """

        :param X_predict:
        :return:
        """

        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """

        :param X_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(X_test)
        return RegressionMeasure.r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"


if __name__ == '__main__':
    regression = LinearRegression()
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50]
    y = y[y < 50]
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

    regression.fit_normal(X_train, y_train)
    print(regression.coef_)
    print(regression.interception_)
    print(regression.score(X_test,y_test))
