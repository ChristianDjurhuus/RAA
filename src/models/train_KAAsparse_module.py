import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import numpy as np
from collections import Counter

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing


class KAAsparse(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, k, data, data2, type = "sparse", seed_init = False,
                 data_type='sparse'):
        super(KAA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Preprocessing.__init__(self, data=data, data_type=data_type, device=self.device, data_2 = None)
        self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
        if seed_init != False:
            np.random.seed(seed_init)
            torch.manual_seed(seed_init)
        Visualization.__init__(self)
        self.edge_list = self.data #TODO: make edgelist here
        self.input_size = (self.N, self.N)
        self.k = k
        self.type = type.lower()
        self.K = self.kernel(self.edge_list)
        self.S = torch.nn.Parameter(torch.randn(self.k, self.N, device = self.device))
        self.C = torch.nn.Parameter(torch.randn(self.N, self.k, device = self.device))
        self.a = torch.nn.Parameter(torch.randn(1, device = self.device))

        self.losses = []

    def kernel(self, edge_list):
        # check pairwise_distances
        degree_diag = torch.diag(torch.unique(edge_list, return_counts=True, device=self.device)[1])
        A = torch.sparse_coo_tensor(edge_list,torch.ones(len(edge_list)), self.N, device=self.device)
        kernel = torch.sparse.mm(torch.sparse.mm((degree_diag**-1), A), torch.sparse.mm((degree_diag**-1), A).T)
        return kernel.float()

    def SSE(self):
        S = torch.softmax(self.S, dim=0)
        C = torch.softmax(self.C, dim=0)
        KC = self.K @ C
        CtKC = C.T @ self.K @ C
        SSt = S @ S.T
        SSE = - 2 * torch.sum( torch.sum( S.T *  KC)) + torch.sum(torch.sum(CtKC * SSt))
        return SSE

    def train(self, iterations, LR = 0.01, print_loss = False):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = self.SSE() / self.N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())