import networkx as nx
import ast
import numpy as np
from util import graph_draw as gw
from sklearn import metrics
import os
from functools import reduce
from util import lfrTools as LFRTool
import json
from FN_yjc import FastNewman as fn
import scipy as sp
import scipy.stats
import math
from yjc_algorithm import Util
# 加载网络
def load_graph(path):
    G = nx.Graph()
    with open(path) as text:
        for line in text:
            vertices = line.strip().split(" ")
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


# 克隆
def clone_graph(G):
    cloned_graph = nx.Graph()
    for edge in G.edges():
        cloned_graph.add_edge(edge[0], edge[1])

    if len(G.nodes()) != len(cloned_graph.nodes()):  # 孤立点没有连边，检查是否有孤立点没有添加进去
        for node in G.nodes():
            if node not in cloned_graph.nodes():
                cloned_graph.add_node(node)

    return cloned_graph


# 计算Q值
def cal_Q(partition, G):
    m = len(list(G.edges()))  # 边的个数
    a = []
    e = []
    # 计算每个社区的a值
    for community in partition:
        t = 0
        for node in community:
            t += len(list(G.neighbors(node)))
        a.append(t / float(2 * m))
    # 计算每个社区的e值
    for community in partition:
        t = 0
        for i in range(len(community)):
            for j in range(len(community)):
                if i != j:
                    if G.has_edge(community[i], community[j]):
                        t += 1
        e.append(t / float(2 * m))
    # 计算Q
    q = 0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

""" 

重叠模块度 EQ

 """
def ExtendQ(G,  coms, edge_num, node_k, node_coms_num):
    print(coms)
    factor = 2.0 * edge_num
    # node_k = dict(sorted(node_k.items(), key=lambda x: x[0], reverse=False))
    first_item = 0.0
    second_item = 0.0
    EQ = 0.0
    for eachc in coms:
        for eachp in coms[eachc]:
            for eachq in coms[eachc]:
                a = 0
                if G.has_edge(eachp,eachq):
                    a = 1
                first_item += a / float(node_coms_num[eachp] * node_coms_num[eachq])
                second_item += node_k[eachp] * node_k[eachq] / float(node_coms_num[eachp] * node_coms_num[eachq])
    EQ = first_item - second_item / factor
    return EQ / factor

""" 

计算划分密度 PD

 """
def cal_pd(G,coms,edge_num):
    pd = 0.0
    dc = 0.0
    for labe, com in coms.items():
        nc = len(com)
        mc = getCommunityEdgesNum(list(com),G)
        # fz = mc - nc +1
        # fm = (nc-2)*(nc-1)
        # if fm != 0:
        #      pd += (mc*fz/fm)
        d_fz = mc - nc +1
        d_fm = nc*(nc-1)/2-(nc-1)
        if d_fm != 0 :
             dc += mc * d_fz / d_fm
    pd = (dc*2)/edge_num
    return pd



# def ExtendQ2(partition, G, node_coms_num):
#     """ 重叠模块度 EQ
#     """
#     print(coms)
#     factor = 2.0 * edge_num
#     # node_k = dict(sorted(node_k.items(), key=lambda x: x[0], reverse=False))
#     first_item = 0.0
#     second_item = 0.0
#     EQ = 0.0
#     for eachc in coms:
#         for eachp in coms[eachc]:
#             for eachq in coms[eachc]:
#                 first_item += A[eachp-1][eachq-1] / float(node_coms_num[eachp] * node_coms_num[eachq])
#                 second_item += node_k[eachp] * node_k[eachq] / float(node_coms_num[eachp] * node_coms_num[eachq])
#     EQ = first_item - second_item / factor
#     return EQ / factor


def communityToArray(array, length, node_param, community_param):#array：社团划分结果，len：节点数；将社团划分结果转换为一维数组
    result = [-1 for x in range(0,length)]

    # try:
    for i in range(len(array)):
        for j in range(len(array[i])):
            value = array[i][j]
            result[value - node_param] = i + community_param
    # except:
    #     print(array
    #     print(length)
    #     print(node_param)
    #     print(community_param)

    return result

def getDataFilePath(name):
    path = "../data/" + name + "/" + name + ".gml"
    if not os.path.exists(path):
        path = "../data/" + name + "/" + name + ".txt"

    return path

def get_graph(name):
    path = getDataFilePath(name)
    if path.endswith(".gml"):
        G = nx.read_gml(path,label='id')
    else:
        G = load_graph(path)
    return G

def getRealFilePath(name):
    return "../data/" + name + "/" + name + "_real.txt"

def getNextWorkXRealResult(name):#获取指定网络的真实划分结果
    filePath = getRealFilePath(name)
    result = []
    with open(filePath, "r") as text:
        line = text.readlines()
        context = str(line[0])
        context = context.replace("\n", "")
        result = ast.literal_eval(context)
    return result

def combainCommunityOnePoint(name, G, result, LFRParam, netType):#合并孤立节点
    delete_index = []
    index = 0
    # for index in range(len(result)):
    while index < len(result) :
        if len(result[index]) == 1:
            #考虑合并
            node = result[index][0]
            max, index_com = getMaxEdge(G, result, node, index)
            result[index_com].append(node)
            result.pop(index)
            # result[index][0] = -1
            # delete_index.append(index)
            index = 0
        else:
            index += 1
    Q = cal_Q(result, G)

    if netType == 0:  # 真实网络和人工网络的NMI计算实现方式不同
        NMI = round(cal_nmi(name, result, G), 4)
    else:  # 人工网络
        NMI = round(LFRTool.LFR_nmi(LFRParam[0], LFRParam[1], result, G), 4)
    return Q, NMI


def combainPoliceCommunityOnePoint(G, result):#合并孤立节点
    delete_index = []
    index = 0
    # for index in range(len(result)):
    while index < len(result) :
        if len(result[index]) == 1:
            #考虑合并
            node = result[index][0]
            max, index_com = getMaxEdge(G, result, node, index)
            result[index_com].append(node)
            result.pop(index)
            # result[index][0] = -1
            # delete_index.append(index)
            index = 0
        else:
            index += 1
    Q = cal_Q(result, G)
    return Q

def getMaxEdge(G, result, node, in_index):#获取某个节点与之连边数最大的社区
    max = -1
    neiber = G.neighbors(node)#node的邻居节点
    index_com = -1
    for index in range(len(result)):
        if index != in_index:
            num_edge = len(list(set(result[index]).intersection(set(neiber))))
            if max < num_edge:
                max = num_edge
                index_com = index
    return max, index_com






def cal_nmi(name, result, graph):#计算NMI，name：网络名称，result：划分结果，graph：图
    param = min(graph.nodes);
    # length = 0;
    # for i in range(len(result)):
    #     length += len(result[i])
    length = len(graph.nodes())
    realResult = getNextWorkXRealResult(name)
    print(realResult)
    node_param = min(graph.nodes);
    community_param = min(realResult)
    print("length:",length)
    A = np.array(communityToArray(result, length, node_param, community_param))#计算结果
    B = np.array(realResult)#真实结果
    # nmiResult = nmiUtil.cal_NMI(A,B)
    nmiResult = metrics.normalized_mutual_info_score(B,A)
    return nmiResult
def over_cal_nmi(name, result, graph):#计算NMI，name：网络名称，result：划分结果，graph：图
    param = min(graph.nodes);
    # length = 0;
    # for i in range(len(result)):
    #     length += len(result[i])
    length = len(graph.nodes())
    communities_real = getNextWorkXRealResult(name)
    real = {}
    for i in range(len(communities_real)):
        if communities_real[i] not in real:
            real[communities_real[i]] = []
        real[communities_real[i]].append(i + 1)
    real_comm = []
    for k, v in real.items():
        real_comm.append(v)
    nmi = Util.onmi(result, real_comm)
    return nmi

def cal_NMIByMember(name, A):
    realResult = getNextWorkXRealResult(name)
    B = np.array(realResult)  # 真实结果
    return metrics.normalized_mutual_info_score(B, A)

def drawResult(G, partition, pos):
    gw.drawCommunity(G, partition, pos)

def getDertQ(partition, G, i, j):#模块度增量
    arr = getDertQArray(partition, G)
    a_i = reduce(lambda x,y:x+y,arr[i])
    a_j = reduce(lambda x,y:x+y,arr[j])
    e_ij = arr[i][j]
    QQ = 2 * (e_ij - (a_i * a_j * 1.0))
    return QQ

def combainCommunity(partition, G, name, netType, LFRParam):
    loction = [-1, -1]#模块度增量最大的两个社区下标位置
    while True:
        maxQQ = -0.1
        k = len(partition)
        for i in range(k):
            for j in range(i + 1, k):
                # cur_QQ = getDertQ2(partition[i], partition[j], G)
                cur_QQ = fn.cal_det_Q(G, partition[i], partition[j])
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

    Q = cal_Q(partition, G)

    if netType == 0:  # 真实网络和人工网络的NMI计算实现方式不同
        NMI = round(cal_nmi(name, partition, G), 4)
    else:  # 人工网络
        NMI = round(LFRTool.LFR_nmi(LFRParam[0], LFRParam[1], partition, G), 4)
    return Q, NMI

def findDivideCommunity(source, target) :#找到被分裂的源社区以及分裂后的社区
    sourceCom = []
    sonCom1 = []
    sonCom2 = []
    for community1 in source:
        find = True
        for index in range(len(target)):
            if len(community1) == len (target[index]) and len(list(set(community1) & set(target[index]))) == len(community1):
                #当前两个社团一样，检查下一个
                find = False
                break
        if find:
            sourceCom = community1
            break
    for node in sourceCom:
        for community in target:
            if node in community:
                if len(sonCom1) == 0:
                    sonCom1 = community
                elif len(sonCom2) == 0 and not len(list(set(sonCom1) & set(community))) == len(community):
                    sonCom2 = community
        if len(sonCom1) != 0 and len(sonCom2) != 0:
            break
    return sourceCom, sonCom1, sonCom2

def getCommunityEdgesNum(community, G):#获得某一个社团内部的边数
    k = len(community)
    sum = 0
    for i in range(k):
        for j in range(i + 1, k):
            node1 = community[i]
            node2 = community[j]
            if (node1, node2) in G.edges() or (node2, node1) in G.edges():
                sum += 1
    return sum

def getTwoCommunityEdgeNum(community_1, community_2,G):#两个社区之间连边数量
    sum = 0
    for node1 in community_1:
        for node2 in community_2:
            if (node1, node2) in G.edges() or (node2, node1) in G.edges():
                sum += 1
    return sum

def getDertQ2(community_1, community_2, G):
    m = len(G.edges())
    e_ij = (getTwoCommunityEdgeNum(community_1, community_2, G)) / (m * 2.0)
    edges_i = 0
    for node in community_1:
        edges_i += G.degree[node]
    edges_i = edges_i - len(community_1)  # 社区j的内部点所关联的边数
    edges_j = 0
    for node in community_2:
        edges_j += G.degree[node]

    edges_j = edges_j - len(community_2)#社区j的内部点所关联的边数
    a_i = float(edges_i) / (m * 2.0)
    a_j = float(edges_j) / (m * 2.0)

    QQ = 2 * (e_ij - a_i * a_j)
    return QQ



def getDertQArray(partition, G):#模块度增量矩阵
    k = len(partition)
    m = len(G.edges())
    array = [[0 for m in range(k)] for n in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            if i == j:
                array[i][j] = 0
            elif array[j][i] != 0:
                array[i][j] = array[j][i]
            else:
                for node1 in partition[i]:
                    for node2 in partition[j]:
                        if (node1, node2) in G.edges() or (node2, node1) in G.edges():
                            array[i][j] += 1
                            array[j][i] += 1
    for i in range(k):
        for j in range(i + 1, k):
            array[i][j] = float(array[i][j]) / (m * 1.0)
            array[j][i] = float(array[j][i]) / (m * 1.0)
    return array

def getCommunityFromJson(filePath):#从json结果中获取社团划分结果
    community_dict = {}
    with open(filePath, "r") as text:
        line = text.readlines()
        # print(line[0])
        community_dict = json.loads(line[0])
        # print(community_dict)

    com_num = max(community_dict.values()) + 1
    # print(com_num)
    membership = [[] for i in range(int(com_num))]
    for node, com in community_dict.items():
        membership[com].append(int(node))
    return membership

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

def getComResult(name, result, graph):

    param = min(graph.nodes);
    length = 0;
    for i in range(len(result)):
        length += len(result[i])
    realResult = getNextWorkXRealResult(name)
    node_param = min(graph.nodes);
    community_param = min(realResult)
    A = np.array(communityToArray(result, length, node_param, community_param))  # 计算结果
    return A

def getPoliceComResult(name, result, graph):

    param = min(graph.nodes);
    length = 0;
    for i in range(len(result)):
        length += len(result[i])
    node_param = min(graph.nodes);
    community_param = 0
    A = np.array(communityToArray(result, length, node_param, community_param))  # 计算结果
    return A


def AllNodesAddOne(list):
    for com_index in range(len(list)):
        for node_index in range(len(list[com_index])):
            list[com_index][node_index] += 1
################## Helper functions ##############
logBase = 2
def __partial_entropy_a_proba(proba):
    if proba==0:
        return 0
    return -proba * math.log(proba,logBase)

def __cover_entropy(cover, allNodes): #cover is a list of set, no com ID
    allEntr = []
    for com in cover:
        fractionIn = len(com)/len(allNodes)
        allEntr.append(sp.stats.entropy([fractionIn,1-fractionIn],base=logBase))

    return sum(allEntr)

def __com_pair_conditional_entropy(cl, clKnown, allNodes): #cl1,cl2, snapshot_communities (set of nodes)
    #H(Xi|Yj ) =H(Xi, Yj ) − H(Yj )
    # h(a,n) + h(b,n) + h(c,n) + h(d,n)
    # −h(b + d, n)−h(a + c, n)
    #a: count agreeing on not belonging
    #b: count disagreeing : not in 1 but in 2
    #c: count disagreeing : not in 2 but in 1
    #d: count agreeing on belonging
    nbNodes = len(allNodes)

    a =len(set(set(allNodes).difference(set(cl))).difference(set(clKnown)))/nbNodes
    b = len(set(clKnown).difference(set(cl)))/nbNodes
    c = len(set(cl).difference(set(clKnown)))/nbNodes
    d = len(set(cl).intersection(set(clKnown)))/nbNodes

    if __partial_entropy_a_proba(a)+__partial_entropy_a_proba(d)>__partial_entropy_a_proba(b)+__partial_entropy_a_proba(c):
        entropyKnown=sp.stats.entropy([len(clKnown)/nbNodes,1-len(clKnown)/nbNodes],base=logBase)
        conditionalEntropy = sp.stats.entropy([a,b,c,d],base=logBase) - entropyKnown
        #print("normal",entropyKnown,sp.stats.entropy([a,b,c,d],base=logBase))
    else:
        conditionalEntropy = sp.stats.entropy([len(cl)/nbNodes,1-len(cl)/nbNodes],base=logBase)
    #print("abcd",a,b,c,d,conditionalEntropy,cl,clKnown)

    return conditionalEntropy #*nbNodes

def __cover_conditional_entropy(cover, coverRef, allNodes, normalized=False): #cover and coverRef and list of set
    X=cover
    Y=coverRef

    allMatches = []
    #print(cover)
    #print(coverRef)
    for com in cover:
        matches = [(com2, __com_pair_conditional_entropy(com, com2, allNodes)) for com2 in coverRef]
        bestMatch = min(matches,key=lambda c: c[1])
        HXY_part=bestMatch[1]
        if normalized:
            HX = __partial_entropy_a_proba(len(com) / len(allNodes)) + __partial_entropy_a_proba((len(allNodes) - len(com)) / len(allNodes))
            if HX==0:
                HXY_part=1
            else:
                HXY_part = HXY_part/HX
        allMatches.append(HXY_part)
    #print(allMatches)
    to_return = sum(allMatches)
    if normalized:
        to_return = to_return/len(cover)
    return to_return


################## Main function ##############


def onmi(cover,coverRef,allNodes=None,variant="LFK"): #cover and coverRef should be list of set, no community ID
    """
    Compute Overlapping NMI
    This implementation allows to compute 3 versions of the overlapping NMI
    LFK: The original implementation proposed by Lacichinetti et al.(1). The normalization of mutual information is done community by community
    MGH: In (2), McDaid et al. argued that the original NMI normalization was flawed and introduced a new (global) normalization by the max of entropy
    MGH_LFK: This is a variant of the LFK method introduced in (2), with the same type of normalization but done globally instead of at each community
    Results are checked to be similar to the C++ implementations by the authors of (2): https://github.com/aaronmcdaid/Overlapping-NMI
    :param cover: set of set of nodes
    :param coverRef:set of set of nodes
    :param allNodes: if、 for some reason you want to take into account the fact that both your cover are partial coverages of a larger graph. Keep default unless you know what you're doing
    :param variant: one of "LFK", "MGH", "MGH_LFK"
    :return: an onmi score
    :Reference:
    1. Lancichinetti, A., Fortunato, S., & Kertesz, J. (2009). Detecting the overlapping and hierarchical community structure in complex networks. New Journal of Physics, 11(3), 033015.
    2. McDaid, A. F., Greene, D., & Hurley, N. (2011). Normalized mutual information to evaluate overlapping community finding algorithms. arXiv preprint arXiv:1110.2515. Chicago
    """
    if (len(cover)==0 and len(coverRef)!=0) or (len(cover)!=0 and len(coverRef)==0):
        return 0
    if cover==coverRef:
        return 1

    if allNodes==None:
        allNodes={n for c in coverRef for n in c}
        allNodes|={n for c in cover for n in c}

    if variant=="LFK":
        HXY = __cover_conditional_entropy(cover, coverRef, allNodes, normalized=True)
        HYX = __cover_conditional_entropy(coverRef, cover, allNodes, normalized=True)
    else:
        HXY = __cover_conditional_entropy(cover, coverRef, allNodes)
        HYX = __cover_conditional_entropy(coverRef, cover, allNodes)

    HX = __cover_entropy(cover, allNodes)
    HY = __cover_entropy(coverRef, allNodes)

    NMI = -10
    if variant=="LFK":
        NMI = 1 - 0.5 * (HXY+ HYX)
    elif variant=="MGH_LFK":
        NMI = 1- 0.5*(HXY/HX+HYX/HY)
    elif variant=="MGH":
        IXY = 0.5*(HX-HXY+HY-HYX)
        NMI =  IXY/(max(HX,HY))
    if NMI<0 or NMI>1 or math.isnan(NMI):
        print("NMI: %s  from %s %s %s %s "%(NMI,HXY,HYX,HX,HY))
        raise Exception("incorrect NMI")
    return NMI

if __name__ == '__main__':
    filePath = "../data/javaCommunity/cora_membership.json"
    community_dict = getCommunityFromJson(filePath)
    com_num = max(community_dict.values()) + 1
    print(com_num)
    membership = [[] for i in range(int(com_num))]
    for node, com in community_dict.items():
        membership[com].append(int(node))

    print(membership)

