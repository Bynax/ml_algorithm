# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/7/5
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class DecisionBoundary:
    @staticmethod
    def plot_decision_boundary(model, axis):
        """

        :param model:
        :param axis:
        :return:
        """
        x0, x1 = np.meshgrid(
            np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
            np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
        )
        X_new = np.c_[x0.ravel(), x1.ravel()]

        y_predict = model.predict(X_new)
        zz = y_predict.reshape(x0.shape)

        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

        plt.contourf(x0, x1, zz, cmap=custom_cmap)
