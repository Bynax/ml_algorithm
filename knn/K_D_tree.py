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
        self._points = np.array(points)
        self._points_dim = np.shape(points)[1]
        self._k = k
        self._kd_tree = None

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
        axis = depth % dim_counts
        points = points[points[:, axis].argsort()]
        root_index = len(points) >> 1
        return {
            "left": KDTree._make_kdtree(points[:root_index], dim_counts, depth + 1),
            "right": KDTree._make_kdtree(points[root_index + 1:], dim_counts, depth + 1),
            "point": points[root_index]
        }

    def build(self):
        """
        构建K-D树
        :return:
        """
        self._kd_tree = KDTree._make_kdtree(self._points, self._points_dim)
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
        assert self._kd_tree is not None, "must build kd-tree before"
        point = np.array(point)
        assert len(point) == self._points_dim, "point's features must be equal to K-D tree features"
        pass

    def get_nearest(self, point):
        """
        _get_nearest的wrapper
        :param point:
        :return:
        """

        assert self._kd_tree is not None, "must build kd-tree before"
        assert len(point) == self._points_dim, "point's features must be equal to K-D tree features"
        return self._get_nearest(self._kd_tree, point, self._points_dim)

    @staticmethod
    def _get_nearest(root, point, dim_counts, depth=0):
        """
        找出kd树中距离给定point最近的结点
        :param root: kd树的根结点
        :param point: 指定结点
        :param dim_counts: 特征数量
        :param depth: 当前的深度，用于决定使用哪一维特征
        :return:
        """
        if root is None:
            return None
        axis = depth % dim_counts
        if point[axis] < root['point'][axis]:
            next_branch = root['left']
            opposit_branch = root['right']
        else:
            next_branch = root['right']
            opposit_branch = root['left']

        candidate = KDTree._get_nearest(next_branch, point, dim_counts, depth + 1)
        best = KDTree._closer_distance(point, candidate, root['point'])

        if KDTree.distance_squared(point, best) > (point[axis] - root['point'][axis]) ** 2:
            candidate = KDTree._get_nearest(opposit_branch, point, dim_counts, depth + 1)
            best = KDTree._closer_distance(point, candidate, best)
        return best

    @staticmethod
    def _closer_distance(pivot, p1, p2):
        """
        找出距离pivot较近的点
        :param pivot: 中心点
        :param p1:
        :param p2:
        :return: 距离pivot较近的点
        """
        if p1 is None:
            return p2

        if p2 is None:
            return p1

        d1 = KDTree.distance_squared(pivot, p1)
        d2 = KDTree.distance_squared(pivot, p2)

        return p1 if d1 < d2 else p2

    @staticmethod
    def distance_squared(point_a, point_b):
        """
        计算给定两样本点的欧氏距离
        :param point_a:
        :param point_b:
        :return: 两个对象的欧氏距离
        """
        # 计算欧式距离的公式，因为这里的距离只是用来比较大小所以省去了np.sqrt的操作
        return np.sum(np.square(np.array(point_a) - np.array(point_b)))


if __name__ == '__main__':
    a = [[3, 6], [2, 7], [17, 15], [6, 12], [13, 15], [9, 1], [10, 19]]
    tree = KDTree(a, 2)
    tree.build()
    print(tree.get_nearest([6, 11]))
