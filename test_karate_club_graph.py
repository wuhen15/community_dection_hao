# -*- coding: UTF-8 -*-
"""
使用空手道数据机karate_club_graph，传入LFM算法，得到communities
构造图的方法有一点差异
"""

"""
@summary: ＬＦＭ算法实现　种子传播算法
@author: shaowenbin

LFM算法首先定义出可以衡量一组节点连接紧密程度的适应度函数：Fitness，具体计算公式如下：
    fg = kgin / (kgin+kgout)α

其中 kgin：表示这些结点内部的度数，也就是内部边的２倍；
    kgout：表示与外部结点相连的度数。一个社区由一组能够使fitness函数最大的结点组成
    也就是再向这个社区中添加任何邻居结点都会使fitness减小。
而一个结点对于这个社区的fitness定义如为包含这个结点的社区的fitness－不包含这个结点的社区的fitness差值：
    fAg = fg+A − fg−A

LFM算法主要由两个步骤构成：选取种子和拓展种子。
    1.它随机地选择一个还没有被分配社区的结点作为种子，通过优化fitness函数的方法拓展它以形成一个社区。
    2.根据适应度函数判断子社团的邻居节点是否可以加入到当前社团当中。
    3.迭代这两步直到所有结点都属于至少一个社区为止。
由于在拓展社区的时候，即使已经被分配社区的结点也可能被添加进来，所以LFM算法是可以发现重叠社区的。

原文链接：https://blog.csdn.net/DreamHome_S/article/details/79849894
"""

import random
# from cdlib import algorithms
import networkx as nx
import matplotlib.pyplot as plt


class Community(object):
    """
    定义扩展的社区
    """

    def __init__(self, graph, alpha=1.0):
        """
        社区属性
        :param graph:
        :param alpha:
        """
        self._graph = graph
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0

    def add_node(self, node):
        """
        子社团中加入节点　改变子社团的出度和入度
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.add(node)
        self._k_in += 2 * node_k_in
        self._k_out = self._k_out + node_k_out - node_k_in

    def remove_vertex(self, node):
        """
        删除节点
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        community_nodes = self._nodes
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2 * node_k_in
        self._k_out = self._k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):
        """
        计算添加节点后适应度的变化
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        """
        计算删除节点后适应度的变化
        :param node:
        :return:
        """
        neighbors = set(self._graph.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        return new_fitness - old_fitness

    def recalculate(self):
        """
        添加节点之后重新计算删除子社团中各个节点的适应度
        :return: node ：to_be_remove
        """
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        """
        返回当前社团的邻居节点
        :return: node ： to_be_examined
        """
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._graph.neighbors(node)) - self._nodes)
        return neighbors


class LFM(object):
    def __init__(self, graph, alpha):
        self._graph = graph
        self._alpha = alpha

    def execute(self):
        """
        种子扩展过程
        :return: communities
        """
        # 划分的社团
        communities = []
        # 未扩展到的节点
        node_not_include = list(self._graph.nodes())

        while len(node_not_include) != 0:
            c = Community(self._graph, self._alpha)
            # 随机选择一个种子节点
            seed = random.choice(node_not_include)

            c.add_node(seed)
            to_be_examined = c.get_neighbors()
            # print(seed)
            # print(to_be_examined)

            # 测试邻居节点中需要扩展的节点
            while to_be_examined:
                # 计算适应度最大的节点进行扩展
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(
                    m.items(), key=lambda x: x[1], reverse=True)[0]

                # 终止条件
                if to_be_add[1] < 0.0:
                    break
                # 当适应度大于等于０时扩展节点
                c.add_node(to_be_add[0])

                # 添加节点之后重新计算删除子社团中各个节点的适应度
                to_be_remove = c.recalculate()
                while to_be_remove is not None:
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            # 删除已经扩展的节点
            for node in c._nodes:
                if node in node_not_include:
                    node_not_include.remove(node)

            # 返回已经完全扩展的社团
            communities.append(c._nodes)

        return communities

"""
//        Properties hbaseProperties = loadProperties("/all-hbase-site.xml");
        Cluster cluster1 = HBaseUtil.getCluster(cluster);
        String zkRoot = cluster1.getZkRoot();
        String zkQuorum = cluster1.getZkQuorum();
//        String parent = hbaseProperties.getProperty(cluster + ".zookeeper.znode.parent");
//        String quorum = hbaseProperties.getProperty(cluster + ".hbase.zookeeper.quorum");
//        String clientPort = hbaseProperties.getProperty(
//                cluster + ".hbase.zookeeper.property.clientPort");
        if (StringUtils.isBlank(zkRoot) || StringUtils.isBlank(zkQuorum)) {
            LOG.error("empty hbase config for cluster : {}", cluster);
            throw new IllegalArgumentIOException("empty hbase config for cluster : " + cluster);
        }
        conf.set("zookeeper.znode.parent", zkRoot);
        conf.set("hbase.zookeeper.quorum", zkQuorum);
        conf.set("hbase.zookeeper.property.clientPort", "2181");
        
        
        //        Configuration conf = HBaseConfiguration.create();
        Cluster cluster1 = HBaseUtil.getCluster(cluster);
        Configuration conf = cluster1.getHDFSConfiguration();
        addJobConf(conf);
        Properties properties = loadProperties("/all-hdfs-site.xml");
        String clusterAlias = properties.getProperty(cluster + ".nameservice.alias");
        conf.set(DFS_NAMESERVICE_KEY, clusterAlias);
        conf.set("fs.defaultFS", "hdfs://" + clusterAlias);
        conf.set("dfs.client.failover.proxy.provider." + clusterAlias,
                properties.getProperty("dfs.client.failover.proxy.provider." + cluster));
        conf.set("dfs.ha.automatic-failover.enabled." + clusterAlias,
                properties.getProperty("dfs.ha.automatic-failover.enabled." + cluster));
        conf.set("dfs.ha.namenodes." + clusterAlias,
                properties.getProperty("dfs.ha.namenodes." + cluster));
        String[] nameNodes = properties.getProperty("dfs.ha.namenodes." + cluster).split(",");
        for (String nameNode : nameNodes) {
            conf.set("dfs.namenode.rpc-address." + clusterAlias + "." + nameNode,
                    properties.getProperty("dfs.namenode.rpc-address." + cluster + "." + nameNode));
            conf.set("dfs.namenode.servicerpc-address." + clusterAlias + "." + nameNode,
                    properties.getProperty("dfs.namenode.servicerpc-address." + cluster + "." + nameNode));
            conf.set("dfs.namenode.http-address." + clusterAlias + "." + nameNode,
                    properties.getProperty("dfs.namenode.http-address." + cluster + "." + nameNode));
            conf.set("dfs.namenode.https-address." + clusterAlias + "." + nameNode,
                    properties.getProperty("dfs.namenode.https-address." + cluster + "." + nameNode));
        }
        loadLocalHbaseResource(conf, cluster);
        return conf;
"""

if __name__ == "__main__":
    # path = "/home/dreamhome/network-datasets/karate/karate.paj"
    # graph = get_graph.read_graph_from_file(path)
    graph=nx.karate_club_graph()
    # graph=graph.to_undirected()
    # for i in range(len(graph.nodes)):
    #     print(graph.nodes[i]['club'])
    print(graph)
    print(graph.nodes[30]['club'])
    print(graph.nodes[30])

    plt.subplot(1, 1, 1)
    nx.draw(graph, with_labels=True)
    plt.title('karate_club_graph')
    plt.axis('on')
    # plt.xticks([])
    # plt.yticks([])

    plt.show()

    algorithm = LFM(graph, 0.8)
    partitions = algorithm.execute()
    print("输出社团划分结果：")
    for i,c in enumerate(partitions):
        print(i+1,sorted(c))