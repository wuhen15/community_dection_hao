import csv
import networkx as nx
import numpy as np
# from util import graphTools as gt
from sklearn import metrics


def getCommunityPath(LFR_path, N, MU):
    path = "D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code/NetworkWithGroundTruth/LFR/" + str(
        LFR_path) + "/N=" + str(N) + "/mu=" + str(MU) + "/community.dat"
    return path


def getNetworkPath(LFR_path, N, MU):
    path = "D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code/NetworkWithGroundTruth/LFR/" + str(
        LFR_path) + "/N=" + str(N) + "/mu=" + str(MU) + "/network.dat"
    return path


def getNetwork(LFR_path, N, MU):
    path = getNetworkPath(LFR_path, N, MU)
    G = load_graph_dat(path)
    return G


def getCommunity(path, G):
    # 获取LFR真实划分
    real_position = [-1 for n in range(len(G.nodes()))]
    with open(path) as text:
        reader = csv.reader(text, delimiter="\t")
        for line in reader:
            source = int(line[0])
            target = int(line[1])
            real_position[source - 1] = target
    return real_position


def communityToArray(array, length, node_param, community_param):
    # array：社团划分结果，len：节点数；将社团划分结果转换为一维数组
    result = [-1 for x in range(0, length)]

    # try:
    for i in range(len(array)):
        for j in range(len(array[i])):
            value = array[i][j]
            result[value - node_param] = i + community_param
    # except:
    #     print(array)
    #     print(length)
    #     print(node_param)
    #     print(community_param)

    return result


# def getCommunity(N, MU):
#     com_path = getCommunityPath(N, MU)
#     net_apth = getNetworkPath(N,MU)
#     result = gh.get_real(com_path, gh.load_graph(net_apth))
#     return result


def write_result(file_path, msg):
    f = open(file_path, "a")
    f.write(msg)
    f.close


def load_graph_dat(path):
    G = nx.Graph()
    with open(path) as text:
        reader = csv.reader(text, delimiter="\t")
        for line in reader:
            source = int(line[0])
            target = int(line[1])
            G.add_edge(source, target)
    return G


def LFR_nmi(LFR_path, N, MU, result, graph):
    # 计算LFR的NMI，name：网络名称，result：划分结果，graph：图
    param = min(graph.nodes)
    length = 0
    for i in range(len(result)):
        length += len(result[i])

    resultPath = getCommunityPath(LFR_path, N, MU)
    realResult = getCommunity(resultPath, graph)
    node_param = min(graph.nodes)
    community_param = min(realResult)
    A = np.array(communityToArray(result, length, node_param, community_param))
    B = np.array(realResult)
    nmiResult = metrics.normalized_mutual_info_score(B, A)
    return nmiResult
