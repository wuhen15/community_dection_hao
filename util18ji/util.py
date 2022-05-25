import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import ast


# 加载网络
def load_graph(path):
    if ".txt" in path:
        # 可自行读取建网
        # G = nx.Graph()
        # with open(path) as text:
        # 	for line in text:
        # 		vertices = line.strip().split(" ")
        # 		source = int(vertices[0])
        # 		target = int(vertices[1])
        # 		G.add_edge(source, target)

        # 用函数读txt建网
        G = nx.read_edgelist(path, create_using=nx.Graph())
    elif ".gml" in path:
        G = nx.read_gml(path, label='id')
    elif ".csv" in path:
        G = nx.read_edgelist(path, create_using=nx.Graph())
    return G
def load_graph1(path):
    G = nx.Graph()
    with open(path) as text:
        for line in text:
            vertices = line.strip().split(" ")
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


def build_G(path):
    G = nx.Graph()
    edges_list = []
    fp = open(path)
    edge = fp.readline().split()
    while edge:
        if edge[0].isdigit() and edge[1].isdigit():
            edges_list.append((int(edge[0]), int(edge[1])))
        edge = fp.readline().split()
    fp.close()
    G.add_edges_from(edges_list)
    return G


# 克隆
def clone_graph(G):
    cloned_graph = nx.Graph()
    for edge in G.edges():
        cloned_graph.add_edge(edge[0], edge[1])
    return cloned_graph


# def del_duplicate_edge(G):
# 计算Q值
def cal_Q(partition, G):
    m = len(list(G.edges()))
    a = []
    e = []

    # 计算每个社区的a值
    for community in partition:
        t = 0
        for node in community:
            t += len(list(G.neighbors(node)))
        a.append(t / float(2 * m))

    # 计算每个社区的e值
    for community in partition:
        t = 0
        # for node in community:
        # 	for neighbor_node in G.neighbors(node):
        # 		if neighbor_node in community:
        # 			t += 1
        for i in range(len(community)):
            for j in range(len(community)):
                if i != j:
                    if G.has_edge(community[i], community[j]):
                        t += 1
        e.append(t / float(2 * m))

    # 计算Q
    q = 0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)

    return q


# 使用矩阵方式计算Q值
# A: 邻接矩阵
# labels: 每个节点划分的community编号
def cal_Q_mat(A_mat, labels):
    S = pd.get_dummies(labels)  # 类别标签转one-hot
    m = sum(sum(A_mat)) / 2
    k = A_mat.sum(axis=1, keepdims=True)
    B = A_mat - (np.tile(k, (1, len(A_mat))) * np.tile(k.T, (len(A_mat), 1))) / (2 * m)
    Q = 1 / (2 * m) * np.trace(S.T @ B @ S)  # @ 等价于np.matmul点乘
    return Q


# 可视化划分结果
def showCommunity(G, partition, pos):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    labels = {}
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(nodeID) + '$'  # 设置可视化label
            cluster[nodeID] = index  # 节点分区号

    # 可视化节点
    colors = ['r', 'g', 'b', '#FFFF66', '#FF9966', '#99CCFF', '#FF00FF', '#FFCC33', '#99FF33', '#660066', '#993333']
    shapes = ['v', 'D', 'o', '^', 's', '<', '>', 'v', '8', 'p', 'h']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index],
                               node_shape=shapes[index],
                               node_size=350,
                               alpha=1)

    # 可视化边
    edges = {len(partition): []}
    for link in G.edges():
        # cluster间的link
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的link
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)

    for index, edgelist in edges.items():
        # cluster内
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index])
        else:
            # cluster间
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=3, alpha=0.8, edge_color=colors[index])

    # 可视化label
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.axis('off')
    plt.savefig('Fig.png', dpi=300)
    plt.show()


def cal_nmi(graph, communities, communities_real_path):
    param = min(graph.nodes)
    length = 0
    for i in range(len(communities)):
        length += len(communities)
    # 获取真实社团结构
    communities_real = []
    with open(communities_real_path, "r") as text:
        line = text.readlines()
        context = str(line[0])
        context = context.replace("\n", "")
        communities_real = ast.literal_eval(context)

    # community to array
    result = [-1 for x in range(0, len(graph.nodes))]
    for i in range(len(communities)):
        for j in range(len(communities[i])):
            result[communities[i][j] - param] = i + 1

    A = np.array(result)
    B = np.array(communities_real)
    # print("A", A)
    # print("B", B)
    nmiResult = metrics.normalized_mutual_info_score(B, A)
    return nmiResult

def nmi(graph, communities, communities_real):
    param = min(graph.nodes)
    length = 0
    for i in range(len(communities)):
        length += len(communities)





# ##########################################
# 计算两个类别标签的NMI指数
def NMI(A, B):
    # return metrics.normalized_mutual_info_score(A, B) # 0.3646247961942429

    eps = 1E-8
    A_one_hot = pd.get_dummies(A)
    B_one_hot = pd.get_dummies(B)
    Pxy = np.dot(B_one_hot.T, A_one_hot)

    Pxy = Pxy / Pxy.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)

    MI = Pxy * np.log(Pxy / np.dot(Px, Py) + eps)
    Hx = -np.dot(Px.T, np.log(Px + eps))
    Hy = -np.dot(Py, np.log(Py + eps).T)

    return 2 * MI.sum() / (Hx.sum() + Hy.sum())


# 计算两个类别标签的ARI指数
def ARI(A, B):
    return metrics.adjusted_rand_score(A, B)
