from sklearn import metrics
import ast
import util.filepath as fp
import numpy as np

def communityToArray(array, length, param):#array：社团划分结果，len：节点数；将社团划分结果转换为一维数组
    result = [-1 for x in range(0,length)]

    for i in range(len(array)):
        for j in range(len(array[i])):
            result[array[i][j] - param] = i + 1

    return result

def getNextWorkXRealResult(name):#获取指定网络的真实划分结果
    filePath = fp.getRealFilePath(name)
    result = []
    with open(filePath, "r") as text:
        line = text.readlines()
        context = str(line[0])
        context = context.replace("\n", "")
        result = ast.literal_eval(context)
    return result
def cal_nmi(name, result, graph):
    param = min(graph.nodes);
    length = 0;
    for i in range(len(result)):
        length += len(result[i])
    realFileName = getNextWorkXRealResult(name)
    A = np.array(communityToArray(result, length, param))
    B = np.array(getNextWorkXRealResult(name))
    # nmiResult = nmiUtil.cal_NMI(A,B)
    nmiResult = metrics.normalized_mutual_info_score(B,A)
    return nmiResult