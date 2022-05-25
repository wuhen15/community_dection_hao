def ExtendQ(G, coms, edge_num, node_k, node_coms_num):
    # print(coms)
    factor = 2.0 * edge_num
    # node_k = dict(sorted(node_k.items(), key=lambda x: x[0], reverse=False))
    first_item = 0.0
    second_item = 0.0
    EQ = 0.0
    for com in coms:
        for eachp in com:
            for eachq in com:
                a = 0
                if G.has_edge(eachp, eachq):
                    a = 1
                first_item += a / float(node_coms_num[eachp] * node_coms_num[eachq])
                second_item += node_k[eachp] * node_k[eachq] / float(node_coms_num[eachp] * node_coms_num[eachq])
    EQ = first_item - second_item / factor
    return EQ / factor
