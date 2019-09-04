# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/9/3
import numpy as np
from decision_tree.decision_tree import DecisionTree


def bootstrap_sample(X, Y):
    """
    有放回的随机采样
    :param X: 训练样本的特征
    :param Y: 训练样本的label或者real-value
    :return:
    """
    N, M = X.shape
    # replace=True表示一个样本可以被选择多次，利用这个参数实现有放回采样
    idxs = np.random.choice(N, N, replace=True)
    return X[idxs], Y[idxs]


class RandomForest:
    def __init__(self, n_trees, max_depth, n_features, classifier=True, criterion="entropy"):
        """
        随机森林的简单实现
        :param n_trees: 森林包含的决策树数目
        :param max_depth: 每棵树的最大深度
        :param n_features: 所使用的特征
        :param classifier: T for class F for regression
        :param criterion: 分裂所使用的标准（MSE、Entropy、GINI）
        """
        self.trees = []
        self.n_trees = n_trees
        self.n_features = n_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.classifier = classifier

    def fit(self, X, Y):
        """
        通过自采样在训练数据中采样数据并训练相应的决策树
        :param X:
        :param Y:
        :return:
        """
        for _ in range(self.n_trees):
            X_samp, Y_samp = bootstrap_sample(X, Y)
            tree = DecisionTree(n_feats=self.n_features, max_depth=self.max_depth, criterion=self.criterion,
                                classifier=self.classifier)
            tree.fit(X_samp, Y_samp)
            self.trees.append(tree)

    def predict(self, X):
        """
        给定X进行预测
        :param X: 给定待预测数据的样本
        :return:
        """
        tree_preds = np.array([[t._traverse(x, t.root) for x in X] for t in self.trees])
        return self._vote(tree_preds)

    def _vote(self, predictions):
        """
        综合每棵树确定最终的预测结果
        :param predictions: 每棵树的预测结果合集
        :return:
        """
        if self.classifier:
            out = [np.bincount(x).argmax() for x in predictions.T]
        else:
            out = [np.mean(x) for x in predictions.T]
        return np.array(out)
