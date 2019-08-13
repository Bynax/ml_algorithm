# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/8/13
import numpy as np

class HMM:

    def __init__(self, A=None, B=None, pi=None):
        """
        简单的HMM的实现


        Parameters
        -----------------
        :param A: 状态转移矩阵，形状为`(N, N)`
        :param B: 发射矩阵，形状为`(N, V)`
        :param pi: 各个隐状态的先验概率分布，N维行向量

        Attributes
        -----------------
        A : 形状为`(N, N)`的状态转移矩阵
        B : 形状为`(N, V)`的发射矩阵
        N : 隐状态的类型个数，（int）
        V : 可观测变量的类型个数，（int）
        O : 可观测序列
        I : 可观测序列O的个数
        T : 每个观测序列中可观测变量的个数
        """
        # transition matrix
        self.A = A

        # emission matrix
        self.B = B

        # prior probability of each latent state
        self.pi = pi
        if self.pi is not None:
            self.pi[self.pi == 0] = self.eps

        # number of latent state types
        self.N = None
        if self.A is not None:
            self.N = self.A.shape[0]
            self.A[self.A == 0] = self.eps

        # number of observation types
        self.V = None
        if self.B is not None:
            self.V = self.B.shape[1]
            self.B[self.B == 0] = self.eps

        # set of training sequences
        self.O = None

        # number of sequences in O
        self.I = None

        # number of observations in each sequence
        self.T = None

    def _farward(self, obs):
        """
        HMM中evaluation问题中所需要的前向算法的实现
        :param obs: 长度为T的观察序列
        :return:
        """
        T = obs.shape[0]

        # 初始化前向概率矩阵
        forward = np.zeros((self.N, T))

        # 初始化第一列的概率
        ot = obs[0]  # ot表示t时刻的可观测变量的值
        for s in range(self.N):
            forward[s, 0] = np.log(self.pi[s]) + np.log(self.B[s, ot])

        # 进行矩阵运算
        for t in range(1, T):
            ot = obs[t]
            for s in range(self.N):
                forward[s, t] = logsumexp(
                    [
                        forward[s_, t - 1]
                        + np.log(self.A[s_, s])
                        + np.log(self.B[s, ot])
                        for s_ in range(self.N)
                    ]
                )
        return forward

    def _backward(self, obs):
        """
        HMM evaluation问题中的backward方法
        :param obs: 长度为T的观察序列
        :return:
        """

        T = obs.shape[0]
        # 初始化backward
        backward = np.zeros((self.N, T))

        for s in range(self.N):
            backward[s, T - 1] = 0

        for t in reversed(range(T - 1)):
            ot1 = obs[t + 1]
            for s in range(self.N):
                backward[s, t] = logsumexp([
                    np.log(self.A[s, s_])
                    + np.log(self.B[s_, ot1])
                    + backward[s_, t + 1]
                    for s_ in range(self.N)
                ])


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
