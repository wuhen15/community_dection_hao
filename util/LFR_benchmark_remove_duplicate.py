'''
去除重复边
'''

edge_file = './data/LFR_benchmark/network.dat'

# 读取数据，并以tab键分割
data = csv.reader(open(edge_file, 'r'), delimiter='\t')
edges = []
for d in data:
    if((d[1], d[0]) not in edges):
        edges.append[(d[0], d[1])]
# 加边
edges = [(d[0], d[1]) for d in data]
G.add_edges_from(edges)