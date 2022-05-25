import sys

sys.path.append(
    'D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code')
import random
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
from util18ji.onmi import onmi
from util18ji.eq import ExtendQ
import collections
from collections import defaultdict
import csv


class Community(object):
    """
    定义扩展的社区
    """

    def __init__(self, graph, alpha=1.0):
        """
        社区属性
        :param graph:
        :param alpha:
        """
        self._graph = graph
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0

    def add_node(self, node):
        """
        子社团中加入节点　改变子社团的出度和入度
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.add(node)
        self._k_in += 2 * node_k_in
        self._k_out = self._k_out + node_k_out - node_k_in

    def remove_vertex(self, node):
        """
        删除节点
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        community_nodes = self._nodes
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2 * node_k_in
        self._k_out = self._k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):
        """
        计算添加节点后适应度的变化
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        """
        计算删除节点后适应度的变化
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        return new_fitness - old_fitness

    def recalculate(self):
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._graph.neighbors(node)) - self._nodes)
        return neighbors


class LFM(object):
    def __init__(self, graph, alpha):
        self._graph = graph
        self._alpha = alpha

    def execute(self):
        """
        种子扩展过程
        :return:
        """
        # 划分的社团
        communities = []
        # 未扩展到的节点
        node_not_include = self._graph.nodes()

        while len(node_not_include) != 0:
            c = Community(self._graph, self._alpha)
            # 随机选择一个种子节点
            seed = random.choice(node_not_include)
            c.add_node(seed)
            to_be_examined = c.get_neighbors()
            # 测试邻居节点中需要扩展的节点
            while to_be_examined:
                # 计算适应度最大的节点进行扩展
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(
                    m.items(), key=lambda x: x[1], reverse=True)[0]

                # 终止条件
                if to_be_add[1] < 0.0:
                    break
                # 当适应度大于等于０时扩展节点
                c.add_node(to_be_add[0])

                # 添加节点之后重新计算删除子社团中各个节点的适应度
                to_be_remove = c.recalculate()
                while to_be_remove is not None:
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            # 删除已经扩展的节点
            for node in c._nodes:
                if node in node_not_include:
                    node_not_include.remove(node)

            # 返回已经完全扩展的社团
            communities.append(c._nodes)

        return communities


if (__name__ == "__main__"):
    # 重叠人工网络
    # 获取网络路径
    name = "LFR2"
    # list_mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_om = [2, 3, 4, 5, 6]
    for om in list_om:
    # mu = 0.6
        # 获取network路径
        network_path = "D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code/NetworkWithGroundTruth/LFR_algo2/" + name + "/om=" + str(
            om) + "/network.dat"
        # 获取community路径
        community_path = "D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code/NetworkWithGroundTruth/LFR_algo2/" + name + "/om=" + str(
            om) + "/community.dat"
        # 构建图
        G = nx.Graph()
        with open(network_path) as text:
            reader = csv.reader(text, delimiter="\t")
            for line in reader:
                source = int(line[0])
                target = int(line[1])
                G.add_edge(source, target)
        # 获取LFR真实社团划分（om不同）
        real_comms_dict = defaultdict(list)
        with open(community_path) as text:
            reader = csv.reader(text, delimiter="\t")
            for line in reader:
                node = int(line[0])
                labels = line[-1].split()
                for i in range(len(labels)):
                    label = int(labels[i])
                    real_comms_dict[label].append(node)
        real_comm = []
        for k, v in real_comms_dict.items():
            real_comm.append(v)

        algorithm = LFM(G, 0.8)
        communities = algorithm.execute()

        # 计算NMI
        ovnmi = onmi(communities, real_comm)
        # print("ovnmi：", ovnmi)
        # 计算EQ
        # 获取边数
        edges_nums = len(nx.edges(G))
        # 获取节点度
        degree_dict = dict(nx.degree(G))
        # 获取每个节点属于的社团数
        node_coms_num = collections.defaultdict(int)
        for node_id in G.nodes():
            for comm in communities:
                if node_id in comm:
                    node_coms_num[node_id] += 1

        eq = ExtendQ(G, communities, edges_nums, degree_dict, node_coms_num)
        # print("eq：", eq)
        # 输出onmi和eq
        print(name + " om = " + str(om) + " ovNMI = " + str(ovnmi) + " EQ = " + str(eq))
