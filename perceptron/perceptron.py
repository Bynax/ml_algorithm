# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/6/22


import numpy as np


class Perceptron(object):
    """
    单层感知机模型的简单实现
    """

    def __init__(self, nums_input, learning_rate=0.01, nums_iter=10):
        """
        初始化Perceptron对象
        :param nums_inputs: 样本的数量
        :param learning_rate: 学习率
        :param num_iter: 最多迭代次数
        """
        self.learning_rate = learning_rate
        self.nums_iter = nums_iter
        self.nums_input = nums_input
        self.weights = np.zeros(nums_input + 1)  # 因为将bias当作w0，因此weights要不输入多一维
        self._X_train = None
        self._Y_train = None

    def _predict(self, inputs):
        """
        根据输入预测输入所属类别
        :param inputs: 输入
        :return: 所属类别
        """
        # 求一个summation
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def predict(self, x_predict):
        """
        对外的预测接口方法
        :param x_predict: 样本特征
        :return:
        """
        assert self._X_train is not None and self._Y_train is not None  # 确保运行predict方法前fit方法已被执行
        return self._predict(x_predict)

    def fit(self, training_inputs, labels):
        """
        模型的训练方法，根据输入数据更新模型的weights属性
        :param training_inputs: 输入特征
        :param labels: 输出的标签
        :return: self 遵循sklearn设计原则
        """

        # 确保训练样本数量和标签一致
        assert training_inputs.shape[0] == labels.shape[0]

        for _ in range(self.nums_iter):
            for inputs, label in zip(training_inputs, labels):
                prediction = self._predict(inputs)  # 预测inputs的值
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs  # 更新weights
                self.weights[0] += self.learning_rate * (label - prediction)  # 更新bias
        return self


if __name__ == '__main__':
    pass
