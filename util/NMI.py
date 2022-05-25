import scipy as sp
import math


logBase = 2


def partialEntropyAProba(proba):
    if proba == 0:
        return 0
    return -proba * math.log(proba, logBase)


def coverEntropy(cover, allNodes):
    allEntr = []
    for com in cover:
        fractionIn = len(com) / len(allNodes)
        allEntr.append(sp.stats.entropy([fractionIn, 1 - fractionIn], base=logBase))

    return sum(allEntr)


def comPairConditionalEntropy(cl, clKnown, allNodes):
    nbNodes = len(allNodes)

    a = len((allNodes - cl) - clKnown) / nbNodes
    b = len(clKnown - cl) / nbNodes
    c = len(cl - clKnown) / nbNodes
    d = len(cl & clKnown) / nbNodes

    if partialEntropyAProba(a) + partialEntropyAProba(d) > partialEntropyAProba(b) + partialEntropyAProba(c):
        entropyKnown = sp.stats.entropy([len(clKnown) / nbNodes, 1 - len(clKnown) / nbNodes], base=logBase)
        conditionalEntropy = sp.stats.entropy([a, b, c, d], base=logBase) - entropyKnown
    else:
        conditionalEntropy = sp.stats.entropy([len(cl) / nbNodes, 1 - len(cl) / nbNodes], base=logBase)

    return conditionalEntropy


def coverConditionalEntropy(cover, coverRef, allNodes, normalized=False):

    allMatches = []

    for com in cover:
        matches = [(com2, comPairConditionalEntropy(com, com2, allNodes)) for com2 in coverRef]
        bestMatch = min(matches, key=lambda c: c[1])
        HXY_part = bestMatch[1]
        if normalized:
            HX = partialEntropyAProba(len(com) / len(allNodes)) + partialEntropyAProba(
                (len(allNodes) - len(com)) / len(allNodes))
            if HX == 0:
                HXY_part = 1
            else:
                HXY_part = HXY_part / HX
        allMatches.append(HXY_part)

    to_return = sum(allMatches)
    if normalized:
        to_return = to_return / len(cover)
    return to_return


def onmi(cover, coverRef, allNodes=None, variant="LFK"):  # cover and coverRef should be list of set, no community ID
    """
    Compute Overlapping NMI
    This implementation allows to compute 3 versions of the overlapping NMI
    LFK: The original implementation proposed by Lacichinetti et al.(1). The normalization of mutual information is done community by community
    MGH: In (2), McDaid et al. argued that the original NMI normalization was flawed and introduced a new (global) normalization by the max of entropy
    MGH_LFK: This is a variant of the LFK method introduced in (2), with the same type of normalization but done globally instead of at each community
    Results are checked to be similar to the C++ implementations by the authors of (2): https://github.com/aaronmcdaid/Overlapping-NMI
    :param cover: set of set of nodes
    :param coverRef:set of set of nodes
    :param allNodes:
    :param variant:
    :param adjustForChance:
    :return:
    :Reference:
    1. Lancichinetti, A., Fortunato, S., & Kertesz, J. (2009). Detecting the overlapping and hierarchical community structure in complex networks. New Journal of Physics, 11(3), 033015.
    2. McDaid, A. F., Greene, D., & Hurley, N. (2011). Normalized mutual information to evaluate overlapping community finding algorithms. arXiv preprint arXiv:1110.2515. Chicago
    """
    if (len(cover) == 0 and len(coverRef) != 0) or (len(cover) != 0 and len(coverRef) == 0):
        return 0
    if cover == coverRef:
        return 1

    if allNodes is None:
        allNodes = {n for c in coverRef for n in c}
        allNodes |= {n for c in cover for n in c}

    if variant == "LFK":
        HXY = coverConditionalEntropy(cover, coverRef, allNodes, normalized=True)
        HYX = coverConditionalEntropy(coverRef, cover, allNodes, normalized=True)
    else:
        HXY = coverConditionalEntropy(cover, coverRef, allNodes)
        HYX = coverConditionalEntropy(coverRef, cover, allNodes)

    HX = coverEntropy(cover, allNodes)
    HY = coverEntropy(coverRef, allNodes)

    NMI = -10
    if variant == "LFK":
        NMI = 1 - 0.5 * (HXY + HYX)
    elif variant == "MGH_LFK":
        NMI = 1 - 0.5 * (HXY / HX + HYX / HY)
    elif variant == "MGH":
        IXY = 0.5 * (HX - HXY + HY - HYX)
        NMI = IXY / (max(HX, HY))
    if NMI < 0 or NMI > 1 or math.isnan(NMI):
        print("NMI: %s  from %s %s %s %s " % (NMI, HXY, HYX, HX, HY))
        raise Exception("incorrect NMI")
    return NMI