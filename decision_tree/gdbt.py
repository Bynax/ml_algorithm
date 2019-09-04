# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/9/4

import numpy as np

from decision_tree import DecisionTree
from losses import CrossEntropyLoss, MSELoss


def to_one_hot(labels, n_classes=None):
    if labels.ndim > 1:
        raise ValueError("labels must have dimension 1, but got {}".format(labels.ndim))
    N = labels.size
    n_cols = np.max(labels) + 1 if n_classes is None else n_classes
    one_hot = np.zeros((N, n_cols))
    one_hot[np.array(N), labels] = 1.0
    return one_hot


class GradientBoostedDecisionTree:
    def __init__(self, n_iter, max_depth=None, classifier=True, learning_rate=1, loss="crossentropy",
                 step_size="constant"):
        """
        gradient boosted ensemble of decision trees.
        :param n_iter:
        :param max_depth:
        :param classifier:
        :param learning_rate:
        :param loss:
        :param step_size:
        """
        self.loss = loss
        self.weights = None
        self.learners = None
        self.out_dims = None
        self.n_iter = n_iter
        self.base_estimator = None
        self.max_depth = max_depth
        self.step_size = step_size
        self.classifier = classifier
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        """
        Fit the gradient boosted decision trees on a dataset
        :param X:
        :param Y:
        :return:
        """
        if self.loss == "mse":
            loss = MSELoss()
        elif self.loss == "crossentropy":
            loss = CrossEntropyLoss()

        if self.classifier:
            Y = to_one_hot(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y

        N, M = X.shape
        self.out_dims = Y.shape[1]
        self.learners = np.empty((self.n_iter, self.out_dims), dtype=object)
        self.weights = np.ones((self.n_iter, self.out_dims))
        self.weights[1:, :] = self.learning_rate

        # fit the base estimator
        Y_pred = np.zeros((N, self.out_dims))
        for k in range(self.out_dims):
            t = loss.base_estimator()
            t.fit(X, Y[:, k])
            Y_pred[:, k] += t.predict(X)
            self.learners[0, k] = t

        # incrementally fit each learner on the negative gradient of the loss
        for i in range(1, self.n_iter):
            for k in range(self.out_dims):
                y, y_pred = Y[:, k], Y_pred[:, k]
                neg_grad = -1 * loss.grad(y, y_pred)

                t = DecisionTree(classifier=False, max_depth=self.max_depth, criterion="mse")

                t.fit(X, neg_grad)
                self.learners[i, k] = t

                step = 1.0
                h_pred = t.predict(X)

                if self.step_size == "adaptive":
                    step = loss.line_search(y, y_pred, h_pred)

                self.weights[i, k] *= step
                Y_pred[:, k] += self.weights[i, k] * h_pred

    def predict(self, X):
        """
        Use the trained model to classify or predict the example in X
        :param X:
        :return:
        """
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        for i in range(self.n_iter):
            for k in range(self.out_dims):
                Y_pred[:, k] += self.weights[i, k] * self.learners[i, k].predict(X)

        if self.classifier:
            Y_pred = Y_pred.argmax(axis=1)

        return Y_pred
