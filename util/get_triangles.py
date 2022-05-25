import networkx as nx
import os

'''
nx.triangles(G, nodes=None) 找到包含一个节点作为顶点的三角形个数
https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.algorithms.cluster.triangles.html#networkx.algorithms.cluster.triangles
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


if __name__ == '__main__':
    filename = os.path.join(
        os.path.abspath('.'),
        'Data\\test.txt'
        )
    G = createGraph(filename)
    print(nx.triangles(G, 1))
