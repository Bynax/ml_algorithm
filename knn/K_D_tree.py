# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/6/25
import numpy as np


class KDTree(object):
    """
    KD 树的简单实现
    """

    def __init__(self, points, k):
        """
        初始化方法
        :param points: 用于构建kd-tree的点的集合,points包含了dimmersion的信息
        :param k: 代表取最近的几个，同knn中k的意义相同
        """
        self.points = np.array(points)
        self.points_dim = np.shape(points)[1]
        self.k = k
        self.kd_tree = None

    @staticmethod
    def _make_kdtree(points, dim_counts, depth=0):
        """
        构建K-D树
        :param points: 构建树的结点
        :param dim_counts points的特征维度
        :param depth 当前深度，根据depth来决定axis，即以哪一个维度的特征进行划分
        :return:
        """
        if len(points) <= 0:
            return None
        axis = (depth+1) % dim_counts
        points = np.sort(points,axis=axis)
        root_index = len(points) >> 1
        return {
            "left": KDTree._make_kdtree(points[:root_index], depth+1),
            "right": KDTree._make_kdtree(points[root_index + 1:], depth+1),
            "point": points[root_index]
        }

    def build(self):
        """
        构建K-D树
        :return:
        """
        self.kd_tree = KDTree._make_kdtree(self.points, self.points_dim)
        return self

    def get_knn(self, point, k, dist_func, return_distances=True, i=0, heap=None):
        """
        获取距离目标结点最近的k个结点
        :param point: 目标结点
        :param k: 目标结点最近点的个数
        :param dist_func: 距离函数
        :param return_distances:
        :param i:
        :param heap:
        :return:
        """
        assert self.kd_tree is not None, "must build kd-tree before"
        point = np.array(point)
        assert np.shape(point)[1] == self.points_dim, "point's features must be equal to K-D tree features"

        pass

    def get_nearest(self, point, dist_func, return_distances=True, i=0, best=None):
        """

        :param point:
        :param dist_func:
        :param return_distances:
        :param i:
        :param best:
        :return:
        """


if __name__ == '__main__':
    a = [[3, 6], [2, 7], [17, 15], [6, 12], [13, 15], [9, 1], [10, 9]]
    b = KDTree(a,2)
    b.build()
    print(b.kd_tree)
