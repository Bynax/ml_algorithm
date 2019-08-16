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

    def _forward(self, obs):
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
        return backward

    def log_likelihood(self, O):
        """
        给定模型和观测序列，计算观测序列的概率
        Parameters
        ----------
        O : 观测序列
        Returns
        -------
        likelihood : 观测序列的loglikelyhood
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        I, T = O.shape

        if I != 1:
            raise ValueError("Likelihood only accepts a single sequence")

        forward = self._forward(O[0])
        log_likelihood = logsumexp(forward[:, T - 1])
        return log_likelihood

    def fit(self, obs, latent_state_type, obs_type, pi=None, tol=1e-5, verbose=False):
        """
        HMM中的training问题，给定观测序列求HMM中的参数A和B
        :param obs: 观测序列
        :param latent_state_type: 隐状态的类型
        :param obs_type:
        :param pi: 各个隐状态的先验概率
        :param tol: 容忍度，如果两个epoch的差小于这个值，停止训练
        :param verbose: 是否打印训练结果
        :return:
        """
        # 初始化各个向量
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # observations
        self.O = obs

        # number of training examples (I) and their lengths (T)
        self.I, self.T = self.O.shape

        # number of types of observation
        self.V = len(obs_type)

        # number of latent state types
        self.N = len(latent_state_type)

        # Uniform initialization of prior over latent states
        self.pi = pi
        if self.pi is None:
            self.pi = np.ones(self.N)
            self.pi = self.pi / self.pi.sum()

        # Uniform initialization of A
        self.A = np.ones((self.N, self.N))
        self.A = self.A / self.A.sum(axis=1)[:, None]

        # Random initialization of B
        self.B = np.random.rand(self.N, self.V)
        self.B = self.B / self.B.sum(axis=1)[:, None]

        # iterate E and M steps until convergence criteria is met
        step, delta = 0, np.inf
        ll_prev = np.sum([self.log_likelihood(o) for o in self.O])
        while delta > tol:
            gamma, xi, phi = self._Estep()
            self.A, self.B, self.pi = self._Mstep(gamma, xi, phi)
            ll = np.sum([self.log_likelihood(o) for o in self.O])
            delta = ll - ll_prev
            ll_prev = ll
            step += 1

            if verbose:
                fstr = "[Epoch {}] LL: {:.3f} Delta: {:.5f}"
                print(fstr.format(step, ll_prev, delta))

        return self.A, self.B, self.pi

    def _Estep(self):
        """
        Baum-Welch算法中的E-step，在这个步骤中估计的参数为``xi`` 和 ``gamma``
        ``xi[i,j,k]`` 是表示根据现有的参数A和B估计当k步时状态为i，（k+1）步时状态为j的概率
        ``gamma[i,j]`` 表示j时刻从状态i开始的概率
        -------
        gamma : 估计的状态-观测数目矩阵
        xi : 估计的状态转移数目矩阵
        phi : 每个隐状态的先验
        """
        gamma = np.zeros((self.I, self.N, self.T))
        xi = np.zeros((self.I, self.N, self.N, self.T))
        phi = np.zeros((self.I, self.N))

        for i in range(self.I):
            obs = self.O[i, :]  # 分别对训练数据的每条进行处理
            fwd = self._forward(obs)  # 前向算法，对观测序列t时刻负责
            bwd = self._backward(obs)  # 后向算法，对观测序列t+1时刻负责
            log_likelihood = logsumexp(fwd[:, self.T - 1])  #

            t = self.T - 1
            for si in range(self.N):
                gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                phi[i, si] = fwd[si, 0] + bwd[si, 0] - log_likelihood

            for t in range(self.T - 1):
                ot1 = obs[t + 1]
                for si in range(self.N):
                    gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                    for sj in range(self.N):
                        xi[i, si, sj, t] = (
                                fwd[si, t]
                                + np.log(self.A[si, sj])
                                + np.log(self.B[sj, ot1])
                                + bwd[sj, t + 1]
                                - log_likelihood
                        )

        return gamma, xi, phi

    def _Mstep(self, gamma, xi, phi):
        """
        Baum-Welch算法中的M-step
        Parameters
        ----------
        gamma : E-step步中估计的状态-观测矩阵
        xi : E-step中估计的状态转移矩阵
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N)`
            The estimated starting count matrix for each latent state.
        Returns
        -------
        A : 估计的转移矩阵
        B : 估计的发射矩阵
        pi : 估计的每个状态的先验概率
        """

        # initialize the estimated transition (A) and emission (B) matrices
        A = np.zeros((self.N, self.N))
        B = np.zeros((self.N, self.V))
        pi = np.zeros(self.N)

        count_gamma = np.zeros((self.I, self.N, self.V))
        count_xi = np.zeros((self.I, self.N, self.N))

        for i in range(self.I):
            Obs = self.O[i, :]
            for si in range(self.N):
                for vk in range(self.V):
                    if not (Obs == vk).any():
                        #  count_gamma[i, si, vk] = -np.inf
                        count_gamma[i, si, vk] = np.log(None)
                    else:
                        count_gamma[i, si, vk] = logsumexp(gamma[i, si, Obs == vk])

                for sj in range(self.N):
                    count_xi[i, si, sj] = logsumexp(xi[i, si, sj, :])

        pi = logsumexp(phi, axis=0) - np.log(self.I)
        np.testing.assert_almost_equal(np.exp(pi).sum(), 1)

        for si in range(self.N):
            for vk in range(self.V):
                B[si, vk] = logsumexp(count_gamma[:, si, vk]) - logsumexp(
                    count_gamma[:, si, :]
                )

            for sj in range(self.N):
                A[si, sj] = logsumexp(count_xi[:, si, sj]) - logsumexp(
                    count_xi[:, si, :]
                )

            np.testing.assert_almost_equal(np.exp(A[si, :]).sum(), 1)
            np.testing.assert_almost_equal(np.exp(B[si, :]).sum(), 1)
        return np.exp(A), np.exp(B), np.exp(pi)


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)