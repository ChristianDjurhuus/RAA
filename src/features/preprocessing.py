import numpy as np
import torch
from collections import defaultdict
import networkx as nx

class Preprocessing():
    def __init__(self, data, data_type: str, device, data_2 = None):
        self.data = data
        self.data_2 = data_2
        self.data_type = data_type
        self.device = device


    def convert_to_egde_list(self):
        if self.data_type == "Edge list":
            N = len(self.data)
            return self.data, N

        if self.data_type == "Adjacency matrix":
            G = nx.from_numpy_matrix(self.data)
            temp = [x for x in nx.generate_edgelist(G, data=False)]
            N = len(self.data)
            edge_list = np.zeros((2, len(temp)))
            for idx in range(len(temp)):
                edge_list[0, idx] = temp[idx].split()[0]
                edge_list[1, idx] = temp[idx].split()[1]
            edge_list = torch.from_numpy(edge_list).long()
            return edge_list, N, G

        if self.data_type == "gml":
            G = nx.read_gml(self.data)
            label_map = {x: i for i, x in enumerate(G.nodes)}
            G = nx.relabel_nodes(G, label_map)
            N = len(G.nodes())
            temp = [x for x in nx.generate_edgelist(G, data=False)]
            edge_list = np.zeros((2, len(temp)))
            for idx in range(len(temp)):
                edge_list[0, idx] = temp[idx].split()[0]
                edge_list[1, idx] = temp[idx].split()[1]
            if self.device == "cuda:0":
                edge_list = torch.tensor(edge_list).long().cuda()
            else:
                edge_list = torch.tensor(edge_list).long()
            return edge_list, N


    def labels(self):
        return NotImplementedError
