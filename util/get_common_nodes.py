import networkx as nx
import os

'''
nx.common_neighbors(G, u, v)获得图中两结点之间的公共结点
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
    common_nodes = nx.common_neighbors(G, 3, 4)
    print(sorted(common_nodes))
