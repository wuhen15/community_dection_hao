import networkx as nx

'''
计算节点的聚集系数 nx.clustering(G)
https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html#networkx.algorithms.cluster.clustering
'''


G = nx.karate_club_graph()
cc = nx.clustering(G, 0)
# cc = nx.clustering(G, 0)
print(cc)
# print(cc[4])
