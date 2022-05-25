import util.modularity as mod
import util.NMI as nmi
def getTwoCommunityEdgeNum(partition, i, j, G):#两个社区之间连边数量
    sum = 0
    for node1 in partition[i]:
        for node2 in partition[j]:
            if (node1, node2) in G.edges() or (node2, node1) in G.edges():
                sum += 1
    return sum

def getDertQ2(partition, G, i, j):
    m = len(G.edges())
    e_ij = (getTwoCommunityEdgeNum(partition, i, j, G)) / (m * 2.0)
    edges_i = 0
    for node in partition[i]:
        edges_i += G.degree[node]
    edges_i = edges_i - len(partition[i])  # 社区j的内部点所关联的边数
    edges_j = 0
    for node in partition[j]:
        edges_j += G.degree[node]

    edges_j = edges_j - len(partition[j])#社区j的内部点所关联的边数
    a_i = float(edges_i) / (m * 2.0)
    a_j = float(edges_j) / (m * 2.0)

    QQ = 2 * (e_ij - a_i * a_j)
    return QQ

def combainCommunity(partition, G, name):

    loction = [-1, -1]#模块度增量最大的两个社区下标位置

    while True:
        maxQQ = -0.1
        k = len(partition)
        for i in range(k):
            for j in range(i + 1, k):
                cur_QQ = getDertQ2(partition, G, i, j)
                if cur_QQ > maxQQ:
                    maxQQ = cur_QQ
                    loction[0] = i
                    loction[1] = j
        if maxQQ > 0:
            index0 = loction[0]
            index1 = loction[1]#将要合并的两个社区
            partition[index0].extend(partition[index1])#将第1个社区并入第0个社区，第1个社区删除
            del partition[index1]#第1个社区删除
        else:
            break

    # Q = mod.cal_Q(partition, G)
    # NMI = nmi.cal_nmi(name, partition, G)
    # print("合并完成：")
    # print(partition)
    # return Q, NMI