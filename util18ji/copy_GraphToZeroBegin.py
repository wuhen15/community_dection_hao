import networkx as nx
def copy_GraphToZeroBegin(G):
    min_value = min(G.nodes())
    cloned_graph = nx.Graph()
    for edge in G.edges():
        cloned_graph.add_edge(edge[0] - min_value, edge[1] - min_value)

    if len(G.nodes()) != len(cloned_graph.nodes()):  # 孤立点没有连边，检查是否有孤立点没有添加进去
        for node in G.nodes():
            if (node-min_value) not in cloned_graph.nodes():
                cloned_graph.add_node(node-min_value)
    return cloned_graph
def copy_GraphToOneBegin(G):
    min_value = min(G.nodes())
    cloned_graph = nx.Graph()
    if min_value == 1:
        return G
    else:
        for edge in G.edges():

            cloned_graph.add_edge(edge[0] + 1, edge[1] +1)

        if len(G.nodes()) != len(cloned_graph.nodes()):  # 孤立点没有连边，检查是否有孤立点没有添加进去
            for node in G.nodes():
                if (node+1) not in cloned_graph.nodes():
                    cloned_graph.add_node(node+1)
        return cloned_graph