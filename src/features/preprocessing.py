import numpy as np
import torch

class Preprocssing():
    def __init__(self, data1, data_type: str, data2 = None):
        self.data1 = data1
        self.data2 = data2
        self.data_type = data_type

    def convert_to_egde_list(self):
        if self.data_type == "Edge list":
            edge_list = []
            edge_list.append(data1)
            edge_list.append(data2)
            return torch.tensor(edge_list)
        if self.data_type == "Adjacency matrix":
            return NotImplementedError

    def labels(self):
        return NotImplementedError


if __name__ == "__main__":

    from numpy import loadtxt
    data1 = loadtxt("data/raw/cora/sparse_i.txt", delimiter=",", unpack=False)
    data2 = loadtxt("data/raw/cora/sparse_j.txt", delimiter=",", unpack=False)

    prepos = Preprocssing(data1 = data1, data2 = data2, data_type = "Edge list")
    edge_list = prepos.convert_to_egde_list()
    print(edge_list)