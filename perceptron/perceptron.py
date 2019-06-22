# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/6/22
import numpy as np
from matplotlib import rcParams

# set the plot figure size
rcParams["figure.figsize"] = 10, 5


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
        self.weights = np.zeros(nums_input + 1)  # 有bias

    def predict(self, inputs):
        """
        根据输入预测输入所属类别
        :param inputs: 输入
        :return: 所属类别
        """
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def fit(self, training_inputs, labels):
        """
        模型的训练方法，根据输入数据更新模型的weights属性
        :param training_inputs: 输入特征
        :param labels: 输出的标签
        :return:
        """
        for _ in range(self.nums_iter):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
