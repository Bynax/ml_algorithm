# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/9/3
import numpy as np


def bootstrap_sample(X, Y):
    """
    有放回的随机采样
    :param X:
    :param Y:
    :return:
    """
    N, M = X.shape
    # replace=True表示一个样本可以被选择多次，利用这个参数实现有放回采样
    idxs = np.random.choice(N, N, replace=True)
    return X[idxs], Y[idxs]

class RandomForest:
    def __init__(self,n_trees,max_depth,):