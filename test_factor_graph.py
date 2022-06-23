"""
测试：
    图G -->  因子图
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
from util18ji.onmi import onmi
from util18ji.eq import ExtendQ
import collections
from collections import defaultdict
import csv
from cdlib import algorithms


def create_graph(network_path):
    """
    构建图
    :param eg: network_path = "./data/test.txt"
    :return: G
    """
    G = nx.Graph()
    with open(network_path) as text:
        reader = csv.reader(text, delimiter="\t")
        for line in reader:
            source = int(line[0].split()[0])
            target = int(line[0].split()[1])
            G.add_edge(source,target)  # The nodes source and target will be automatically added if they are not already in the graph.
    return G


if (__name__ == "__main__"):
    # 获取network路径
    network_path = "./data/test.txt"

    # 构建图
    G = create_graph(network_path)
