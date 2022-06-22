import sys

sys.path.append(
    'D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code')
import util18ji.filepath as fp
import util18ji.NMI as nmi
from util18ji.onmi import onmi
from util18ji.eq import ExtendQ
import util18ji.util as ut
from networkx.algorithms import community
import networkx as nx
import igraph as ig
from igraph import *
import math
import matplotlib.pyplot as plt
import numpy as np
import pyds
import ast
import cdlib
from cdlib import algorithms
import collections
from collections import defaultdict
import csv


'''
种子扩展结合标签传播算法2-2（重叠）

1. 选取种子
按照算法一中的方式选择种子节点

2. 种子扩展（ref: Overlapping Community Discovery Method Based on Two Expansions of Seeds）
利用最大化适应度函数进行扩展，然后利用隶属度进行分配剩余节点

利用pageRank-Nibble进行扩展（没有创新，plan B）

3. 社团合并

'''

def createGraph(filename):
    # 读取txt、out类型文件，建立图
    G = nx.Graph()
    edges_list = []
    fp = open(filename)
    line = fp.readline()
    while line:
        edge = line.split()
        if edge[0].isdigit() and edge[1].isdigit():
            edges_list.append((int(edge[0]), int(edge[1])))
        line = fp.readline()
    fp.close()

    G.add_edges_from(edges_list)
    return G


class algo2_2():
    def __init__(self, G, alpha):
        self._G = G
        self._alpha = alpha

    
    def execute(self):
        # 种子重要性排序
        nodes_importance = self.cal_nodes_importance()
        # 根据节点重要性对节点排序
        nodes_rank = self.rank_nodes(nodes_importance)
        # 选取种子社团
        seeds_set = self.seeds_select(nodes_importance, nodes_rank)
        # print("获取种子并进行种子扩展...")
        communities = self.seeds_expansion(seeds_set)
        # 社团合并
        post_communities = self.merge_communities(communities)
        return post_communities


if __name__ == '__main__':
    '''
    # 空手道数据(2) 海豚数据集(4 or 2) football数据集(12) polbooks数据集(3)
    path = [
        './NetworkWithGroundTruth/data_raw/out.ucidata-zachary',
        './NetworkWithGroundTruth/data_raw/out.dolphins',
        './NetworkWithGroundTruth/data_raw/football.txt',
        './NetworkWithGroundTruth/data_raw/polbook.txt',
        './NetworkWithGroundTruth/data_raw/riskmap.txt',
        './NetworkWithGroundTruth/data_raw/collaboration.txt'
    ]
    # path_real = [
    #     './NetworkWithGroundTruth/data_real/karate_real.txt',
    #     './NetworkWithGroundTruth/data_real/dolphins_real.txt',
    #     './NetworkWithGroundTruth/data_real/football_real.txt',
    #     './NetworkWithGroundTruth/data_real/polbooks_real.txt',
    #     './NetworkWithGroundTruth/data_real/riskmap_real.txt',
    #     './NetworkWithGroundTruth/data_real/collaboration_real.txt'
    # ]
    path_real = [
        './NetworkWithGroundTruth/data_real/karate_real.txt',
        './NetworkWithGroundTruth/data_real/dolphins_real_2.txt',
        './NetworkWithGroundTruth/data_real/football_real.txt',
        './NetworkWithGroundTruth/data_real/polbooks_real.txt',
        './NetworkWithGroundTruth/data_real/riskmap_real.txt',
        './NetworkWithGroundTruth/data_real/collaboration_real.txt'
    ]

    gid = 0
    print("邻接矩阵成图...", path[gid].split('/')[-1])
    G = createGraph(path[gid])

    # 对比算法
    # communities = []
    # # lfm(2009)
    # res = algorithms.lfm(G, alpha=0.8)
    # # SLPA(2011)
    # res = algorithms.slpa(G, t=21, r=0.1)
    # # demon(2012)
    # res = algorithms.demon(G, min_com_size=3, epsilon=0.25)
    # # ego-splitting(2017)
    # res = algorithms.egonet_splitter(G)
    # node_perception(2015)
    # res = algorithms.node_perception(G, threshold=0.25, overlap_threshold=0.25)
    # 处理cdlib结果
    # for nc in res.communities:
    #     communities.append(nc)

    # 获取真实社团划分
    with open(path_real[gid], "r") as text:
        line = text.readlines()
        context = str(line[0])
        context = context.replace("\n", "")
        communities_real = ast.literal_eval(context)
    real = {}
    for i in range(len(communities_real)):
        if communities_real[i] not in real:
            real[communities_real[i]] = []
        real[communities_real[i]].append(i + 1)
    real_comm = []
    for k, v in real.items():
        real_comm.append(v)

    # # 个人算法
    algo = algo2_2(G, alpha)
    communities = algo.execute()
    print("社团数量：", len(communities))
    # 计算NMI
    ovnmi = onmi(communities, real_comm)
    print("ovnmi：", ovnmi)
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
    print("eq：", eq)
    '''
    # 重叠人工网络
    # 获取网络路径
    name = "LFR2"
    # list_mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_om = [2, 3, 4, 5, 6]
    # list_N = [1000, 2000, 3000, 4000, 5000]
    # list_on = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
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

        # 个人算法
        # algo = algo2_2(G, alpha)
        # communities = algo.execute()

        # 对比算法
        communities = []
        # # lfm(2009)
        res = algorithms.lfm(G, alpha=1)
        # # SLPA(2011)
        res = algorithms.slpa(G, t=21, r=0.1)
        # # demon(2012)
        # res = algorithms.demon(G, min_com_size=3, epsilon=0.25)
        # # ego-splitting(2017)
        # res = algorithms.egonet_splitter(G)
        # node_perception(2015)
        # res = algorithms.node_perception(G, threshold=0.25, overlap_threshold=0.25)
        # 处理cdlib结果
        for nc in res.communities:
            communities.append(nc)

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
