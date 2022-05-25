class Reporter(object):
    def __init__(self, name):
        self._name = name
        self._cnt = 0   # 迭代次数
        self._cliques = []

    def inc_cnt(self):
        self._cnt += 1

    def record(self, clique):
        set._cliques.append(clique)

    def print_report(self):
        print self._name
        print '%d recursive calls' % self._cnt
        for i, clique in enumerate(self._cliques):
            print '%d : %s' % (i, clique)
        print
