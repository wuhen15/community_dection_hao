
import csv
import math
from yjc_algorithm import Util
import networkx as nx
import numpy as np
from util import graphTools as gt
from sklearn import metrics
from collections import defaultdict

def getCommunityPath(N,MU):
    path = "../data/LFR/N=" + str(N) + "/mu=" + str(MU) + "/community.dat"
    return path

def getCommunityPath_du(N,MU):
    path = "../du/N=" + str(N) + "/mu=" + str(MU) + "/community.dat"
    return path

def getNetworkPath(N,MU):
    path = "../data/LFR/N=" + str(N) + "/mu=" + str(MU) + "/network.dat"
    return path

def getNetworkPath_du(N,MU):
    path = "../du/N=" + str(N) + "/mu=" + str(MU) + "/network.dat"
    return path


def getNetwork_du(N, MU):
    path = getNetworkPath_du(N, MU)
    G = load_graph_dat(path)
    return G

def getNetwork(N, MU):
    path = getNetworkPath(N, MU)
    G = load_graph_dat(path)
    return G

def getCommunity(path, G):#获取LFR真实划分
    real_position = [-1 for n in range(len(G.nodes()))]
    with open(path) as text:
        reader = csv.reader(text, delimiter="\t")
        for line in reader:
            source = int(line[0])
            # print("line 0:",line[0])
            # print("line 1:", line[1])
            # for
            # print("line 2:", line[2])
            target = int(line[1])
            real_position[source - 1] = target
    print("real_position",real_position)
    return real_position

def getOverlappingCommunity(path):#获取重叠LFR真实划分
    real_dict = defaultdict(list)
    with open(path) as text:
        reader = csv.reader(text, delimiter="\t")
        index = 0
        for line in reader:
            tmp_str = line[-1].split()
            node = int(line[0])
            if len(tmp_str) == 2:
                 labe_1 = int(tmp_str[0])
                 labe_2 = int(tmp_str[1])
                 real_dict[labe_1].append(node)
                 real_dict[labe_2].append(node)
            else:
                  real_dict[int(line[1])].append(node)
    real_list = []
    for k, v in real_dict.items():
        real_list.append(v)
    print("real_list",real_list)
    return real_list

# def getCommunity(N, MU):
#     com_path = getCommunityPath(N, MU)
#     net_apth = getNetworkPath(N,MU)
#     result = gh.get_real(com_path, gh.load_graph(net_apth))
#     return result

def write_result(file_path,msg):
    f=open(file_path,"a")
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

def LFR_nmi(N, MU, result, graph):#计算LFR的NMI，name：网络名称，result：划分结果，graph：图
    param = min(graph.nodes);
    length = 0;
    for i in range(len(result)):
        length += len(result[i])
    resultPath = getCommunityPath(N, MU)
    realResult = getCommunity(resultPath,graph)
    # print(realResult)
    node_param = min(graph.nodes);
    community_param = min(realResult)
    A = np.array(gt.communityToArray(result, length, node_param, community_param))
    B = np.array(realResult)
    nmiResult = metrics.normalized_mutual_info_score(B,A)
    return nmiResult
class OverlapNMI:
    @staticmethod
    def entropy(num):
        return -num * math.log(2, num)

    def __init__(self, num_vertices, result_comm_list, ground_truth_comm_list):
        self.x_comm_list = result_comm_list
        self.y_comm_list = ground_truth_comm_list
        self.num_vertices = num_vertices

    def calculate_overlap_nmi(self):
        def get_cap_x_given_cap_y(cap_x, cap_y):
            def get_joint_distribution(comm_x, comm_y):
                prob_matrix = np.ndarray(shape=(2, 2), dtype=float)
                intersect_size = float(len(set(comm_x) & set(comm_y)))
                cap_n = self.num_vertices + 4
                prob_matrix[1][1] = (intersect_size + 1) / cap_n
                prob_matrix[1][0] = (len(comm_x) - intersect_size + 1) / cap_n
                prob_matrix[0][1] = (len(comm_y) - intersect_size + 1) / cap_n
                prob_matrix[0][0] = (self.num_vertices - intersect_size + 1) / cap_n
                return prob_matrix

            def get_single_distribution(comm):
                prob_arr = [0] * 2
                prob_arr[1] = float(len(comm)) / self.num_vertices
                prob_arr[0] = 1 - prob_arr[1]
                return prob_arr

            def get_cond_entropy(comm_x, comm_y):
                prob_matrix = get_joint_distribution(comm_x, comm_y)
                entropy_list = list(map(OverlapNMI.entropy,
                                   (prob_matrix[0][0], prob_matrix[0][1], prob_matrix[1][0], prob_matrix[1][1])))
                if entropy_list[3] + entropy_list[0] <= entropy_list[1] + entropy_list[2]:
                    return np.inf
                else:
                    prob_arr_y = get_single_distribution(comm_y)
                    return sum(entropy_list) - sum(list(map(OverlapNMI.entropy, prob_arr_y)))

            partial_res_list = []
            for comm_x in cap_x:
                cond_entropy_list = list(map(lambda comm_y: get_cond_entropy(comm_x, comm_y), cap_y))
                min_cond_entropy = float(min(cond_entropy_list))
                partial_res_list.append(
                    min_cond_entropy / sum(list(map(OverlapNMI.entropy, get_single_distribution(comm_x)))))
            return np.mean(partial_res_list)

        return 1 - 0.5 * get_cap_x_given_cap_y(self.x_comm_list, self.y_comm_list) - 0.5 * get_cap_x_given_cap_y(
            self.y_comm_list, self.x_comm_list)

def OverLapping_LFR_nmi(N, MU, nums_v,result,graph):#计算LFR的NMI，name：网络名称，result：划分结果，graph：图
    resultPath = getCommunityPath(N, MU)
    realResult = getOverlappingCommunity(resultPath)
    nmi = Util.onmi(result,realResult)
    # omni = OverlapNMI(nums_v,result,realResult)
    return nmi

def LFR_nmi_du(N, MU, result, graph):#计算LFR的NMI，name：网络名称，result：划分结果，graph：图
    param = min(graph.nodes);
    length = 0;
    for i in range(len(result)):
        length += len(result[i])

    resultPath = getCommunityPath_du(N, MU)
    realResult = getCommunity(resultPath,graph)
    node_param = min(graph.nodes);
    community_param = min(realResult)
    A = np.array(gt.communityToArray(result, length, node_param, community_param))
    B = np.array(realResult)
    nmiResult = metrics.normalized_mutual_info_score(B,A)
    return nmiResult



