import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (11, 9), (9, 7), (10, 8), (8, 6),  # arms
    (16, 14), (14, 12), (17, 15), (15, 13),  # legs
    (12, 6), (13, 7), (12, 13), (6, 7),  # torso
    (6, 1), (7, 1), (2, 1), (3, 1), (4, 2), (5, 3)  # nose, eyes and ears
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
