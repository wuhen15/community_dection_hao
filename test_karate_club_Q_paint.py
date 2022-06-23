"""
@summary: karate-club空手道成员俱乐部
社区模块度计算和节点着色
https://blog.csdn.net/just_so_so_fnc/article/details/107337931
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# 获取空手道俱乐部成员数据
def getKarateClub():
    G = nx.karate_club_graph()
    G = G.to_undirected()
    return G


# 构造邻接矩阵和度矩阵
def karateMatrix(G):
    N = len(G.adj)
    # 初始化邻接矩阵
    A = np.zeros((N, N))
    # 保存节点所在社区标记
    C = []
    # 保存同一社区节点的颜色
    colors = []
    for i in range(len(G.adj)):
        # 若节点‘club’值为officer，则为1社区，颜色为#377eb8
        if G.nodes[i]['club'] == "Officer":
            C.append(1)
            colors.append('#377eb8')
        else:
            C.append(0)
            colors.append('#ff7f00')
        # 获取i节点的邻接点，转化为list格式保存
        nodeList = list(G.adj[i])
        # 存在边 邻接矩阵中值为1
        for j in nodeList:
            A[i][j] = 1

    return A, C, colors


# 计算模块度
def getModurality(A, C):
    # 通过邻接矩阵计算每一个节点的度
    digList = np.sum(A, axis=1)
    # 整个网络中所有节点的度
    digs = np.sum(digList)
    # 保存最终整个网络的模块度
    Q = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if C[i] == C[j]:
                c = 1
            else:
                c = 0
            # 任意两个节点间的计算方式，然后累加
            q = (1 / digs) * (A[i][j] - (digList[i] * digList[j]) / digs) * c
            Q += q
    return Q


# 作图
def plot(G, colors):
    plt.subplot(111)
    # node_color根据colors列表中的颜色确定
    nx.draw(G, with_labels=True, font_weight='bold', node_color=colors)
    plt.show()


# 主函数
def main():
    G = getKarateClub()
    A, C, colors = karateMatrix(G)
    Q = getModurality(A, C)
    plot(G, colors)
    print("the modurality of the community is:", Q)  # 最后输出0.358


if __name__ == '__main__':
    main()


