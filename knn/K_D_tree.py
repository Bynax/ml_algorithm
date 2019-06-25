# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/6/25

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
        self.points = points
        self.k = k
        self.kd_tree = None

    def _build_kdtree(self, depth=0):
        """
        构建KD-tree
        :param depth:当前深度，因为要根据深度使用不同的feature进行划分
        :return:
        """
        n = len(points)

        if n <= 0:
            return None

        axis = depth % k

        sorted_points = sorted(points, key=lambda point: point[axis])

        return {
            'point': sorted_points[n / 2],
            'left': build_kdtree(sorted_points[:n / 2], depth + 1),
            'right': build_kdtree(sorted_points[n / 2 + 1:], depth + 1)
        }

    def _get_parent(self, node):
        """
        获得指定结点的父结点
        :param node: 需要寻找父结点的结点
        :return: 返回该结点的副结点
        """
        pass

    def _delete_node(self, node):
        """
        删除结点函数
        :param node: 需要删除的结点
        :return: self
        """
        return self

    def search_knn(self,node):
        """
        返回指定结点
        :param node:
        :return:
        """
