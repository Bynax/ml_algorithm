# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019-08-30
import numpy as np


class Node:

    def __init__(self, left, right, rule):
        """

        :param left: left tree
        :param right: right tree
        :param rule:
        """
        self.left = left
        self.right = right
        self.feature = rule[0]
        self.threshold = rule[1]


class Leaf:
    def __init__(self, value):
        """
        :param value:
        """
        self.value = value


class DecisionTree:

    def __init__(self, classifier=True, max_depth=None, n_feats=None, criterion="entropy", seed=None):
        """

        :param classifier:
        :param max_depth:
        :param n_feats:
        :param criterion:
        :param seed:
        """
        if seed:
            np.random.seed(seed)
        self.depth = 0
        self.root = None
        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf

        if not classifier and criterion in ["gini", "entroy"]:
            raise ValueError(
                "{} is a valid criterion only when classifier=True".format(criterion)
            )
        if classifier and criterion == "mse":
            raise ValueError("'mse' is valid criterion only for classifier=False")

    def fit(self, X, Y):
        """
        Fit a binary decision tree to a dataset
        :param X: training data
        :param Y: An array of class lables for each example if classifier=True,
                  otherwise the set of target value of examples.
        :return:
        """
        self.n_classes = max(Y) + 1 if self.classifier else None  # 有几个类
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])  # 有多少个特征
        self.root = self._grow(X, Y)

    def predict(self, X):
        """
        对于给定的X进行预测
        :param X:
        :return:
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_prob(self, X):
        """

        :param X:
        :return:
        """
        assert self.classifier, "'predict_class_prob' is not defined for classifier = False"
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y):
        """
        构造决策树
        :param X: 数据的特征向量
        :param Y: 数据的标签
        :return: 决策树构造后的根节点
        """
        if (len(set(Y)) == 1):  # 弱纯度达到最大，返回叶子节点
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
                return Leaf(prob)
            return Leaf(Y[0])
        if self.depth >= self.max_depth:  # 若达到最大高度，返回叶子节点
            v = np.mean(Y, axis=0)  # 对于回归树，返回该节点上所有值的均值
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)  # 对于分类树，返回概率值
            return Leaf(v)

        N, M = X.shape  # N->num M->features
        self.depth += 1  # 更新树的高度
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)  #

        feat, thresh = self._segment(X, Y, feat_idxs)
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        left = self._grow(X[l, :], Y[l])
        right = self._grow(X[r, :], Y[r])
        return Node(left, right, (feat, thresh))

    def _segment(self, X, Y, feat_idxs):
        """
        给定训练样本，给定数据的特征，对于每一个特征寻找threshold
        看哪个特征可以将样本分割得更好，就选用该样本作为分割的标准
        :param X: 样本特征
        :param Y: 样本标签
        :param feat_idxs: 特征集合
        :return:
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[:-1] + levels[1:]) / 2
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]

        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """

        :param Y:
        :param split_thresh:
        :param feat_values:
        :return:
        """
        if self.criterion == "entropy":
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse

        parent_loss = loss(Y)

        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = entropy(Y[left]), entropy(Y[right])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_loss - child_loss
        return ig

    def _traverse(self, X, node, prob=False):
        """

        :param X:
        :param node:
        :param prob:
        :return:
        """
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)


def mse(y):
    """
    calculate the mse
    :param y:
    :return:
    """
    return np.mean((y - np.mean(y)) ** 2)


def entropy(y):
    """
    calculate the entropy
    :param y:
    :return:
    """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    """
    calculate the gini index
    :param y:
    :return:
    """
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])
