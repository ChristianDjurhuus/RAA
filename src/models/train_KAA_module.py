import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.metrics import jaccard_score
from torch_sparse import spspmm
import numpy as np

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing


class KAA(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, k, data, type = "jaccard", link_pred = False, test_size = 0.3):
        super(KAA, self).__init__()
        self.link_pred = link_pred
        self.X_test = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Preprocessing.__init__(self, data=data, data_type='adjacency matrix', device=self.device, data_2 = None)
        self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
        Visualization.__init__(self)

        self.test_size = test_size
        Link_prediction.__init__(self)
        self.X = self.data
        self.input_size = (self.N, self.N)
        self.k = k
        self.type = type.lower()
        self.K = self.kernel(self.X, type = self.type)
        self.S = torch.nn.Parameter(torch.randn(self.k, self.N, device = self.device))
        self.C = torch.nn.Parameter(torch.randn(self.N, self.k, device = self.device))
        self.a = torch.nn.Parameter(torch.randn(1, device = self.device))

        self.losses = []

    def kernel(self, X, type):
        # check pairwise_distances
        #kernel = X.T@X
        if type == 'jaccard':
            kernel = 1-torch.from_numpy(pairwise_distances(X.T, X, metric=type)).float()
        if type == 'parcellating': #TODO: Does not seem to learn the structure.
            temp = ((X.unsqueeze(1) - X + 1e-06)**2).sum(-1)
            kernel = (2 * (temp - torch.diag(torch.diagonal(temp))))**0.5
        '''if type == 'normalised_x':
            kernel = X @ X.T / (X @ X.T).sum(0) #TODO: Sum row or column wise?'''
        if type == 'laplacian':
            D = torch.diag(X.sum(1))
            kernel = D - X #TODO: weird space..
        return kernel

    def SSE(self):
        S = torch.softmax(self.S, dim=0)
        C = torch.softmax(self.C, dim=0)
        KC = self.K @ C 
        CtKC = C.T @ self.K @ C
        SSt = S @ S.T
        SSE = - 2 * torch.sum( torch.sum( S.T *  KC)) + torch.sum(torch.sum(CtKC * SSt))
        return SSE

    def train(self, iterations, LR = 0.01, print_loss = True):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = self.SSE() / self.N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())
