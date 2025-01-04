import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import scipy.io as scio
import pandas as pd
import numpy as np
import warnings
import xlsxwriter
warnings.simplefilter(action='ignore')
import scipy.io as scio
from matplotlib import pyplot as plt
import numpy as np
import xlwt
from networkx import spring_layout


class Graph_Matrix:
    def __init__(self, vertices=[], matrix=[]):

        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):

        if tail not in self.vertices:
            self.add_vertex(tail)

        if head not in self.vertices:
            self.add_vertex(head)

        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))



####由邻接矩阵生成属性拓扑图
def create_undirected_matrix(my_graph,chuanshu):
    nodes = ['0°-20°', '20°-40°', '40°-60°', '60°-80°', '80°-100°', '100°-120°', '120°-140°', '140°-160°','160°-180°']
    matrix = chuanshu
    my_graph = Graph_Matrix(nodes,matrix)
    graph_bian =my_graph.getAllEdges()
    return my_graph,graph_bian



def draw_undircted_graph(my_graph):
    G = nx.Graph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(str(node))

    for edge in my_graph.edges:
        G.add_edge(str(edge[0]), str(edge[1]), weight = str(edge[2]))   #, weight = str(edge[2])

    return G

# 深度优先搜索(DFS)
def dfs(graph, v):
    visited.append(v)
    stack.append(v)
    while stack:
        node = stack.pop()
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.append(neighbour)
                stack.append(neighbour)

workbook = xlsxwriter.Workbook(r'F:\shiyan\new_attempt\SPDD\1yupu_AT/SPDD患病样本中AT图中的连通域数量.xls')
worksheet = workbook.add_worksheet('sheet1')
fea = 1

for path in os.listdir(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\lianjiematrix\PD'):
    data=scio.loadmat(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\lianjiematrix\PD'+ "./" + path)
    x=data['1']   #病人1   健康人0
    my_graph = Graph_Matrix()   #实例化类
    print(int(np.shape(x)[0] / 9) + 1)

    for i in range(1, int(np.shape(x)[0] / 9) + 1):
        # print("第几个--------------")
        # print(i)
        matrix = x[(9 * (i - 1)):(9 * i), :]
        # print(matrix)
        for m in range(9):
            for n in range(9):
                if m == n:
                    matrix[m][n] = 0
        created_graph, created_weight = create_undirected_matrix(my_graph, matrix)
        matrix1=draw_undircted_graph(created_graph)
        liantong_jishu = 0
        visited = []
        stack = []
        for v in matrix1:
            if v not in visited:
                liantong_jishu = liantong_jishu + 1
                dfs(matrix1, v)

        print(liantong_jishu)
        worksheet.write(fea, i, liantong_jishu)
    fea = fea + 1

workbook.close()





