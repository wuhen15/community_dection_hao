def AllNodesAddOne(list):
    for com_index in range(len(list)):
        for node_index in range(len(list[com_index])):
            list[com_index][node_index] += 1