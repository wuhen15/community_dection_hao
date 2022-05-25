import networkx as nx
import networkx.algorithms.clique as C

if __name__ == "__main__":
    # G = nx.karate_club_graph()
    G = nx.Graph()
    G.add_nodes_from(set(range(1, 8)))
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (5, 6), (5, 7), (6, 7)])
    cliques = C.find_cliques(G)
    for clique in cliques:
        print(clique)
