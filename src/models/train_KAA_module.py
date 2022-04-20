import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.metrics import jaccard_score
from torch_sparse import spspmm

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing


class KAA(nn.Module):
    def __init__(self, k, data, type = "jaccard", data_type = "Edge list", data_2 = None):
        super(KAA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Preprocessing.__init__(self, data = data, data_type = data_type, device = self.device, data_2 = data_2)
        #self.edge_list, self.N = Preprocessing.convert_to_egde_list(self)
        Link_prediction.__init__(self, edge_list = self.edge_list)
        Visualization.__init__(self)

        self.X = data
        self.N = self.X.shape[0]
        self.input_size = (self.N, self.N)
        self.k = k
        self.type = type
        self.K = self.kernel(type = self.type)
        self.S = torch.nn.Parameter(torch.randn(self.k, self.input_size[0], device = self.device))
        self.C = torch.nn.Parameter(torch.randn(self.input_size[0], self.k, device = self.device))
        self.a = torch.nn.Parameter(torch.randn(1, device = self.device))

        self.losses = []


    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx = torch.multinomial(self.sampling_weights, self.sample_size, replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator = torch.cat([sample_idx.unsqueeze(0), sample_idx.unsqueeze(0)], 0)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0) #.to(self.device)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1], device = self.device), indices_translator,
                                torch.ones(indices_translator.shape[1], device = self.device), self.input_size[0], self.input_size[0],
                                self.input_size[0], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_translator, torch.ones(indices_translator.shape[1], device = self.device), indexC, valueC,
                                self.input_size[0], self.input_size[0], self.input_size[0], coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_idx, sparse_i_sample, sparse_j_sample

    def kernel(self, type):
        #type: #check pairwise_distances...
        #kernel = X.T@X
        if type == 'jaccard':
            kernel = 1-torch.from_numpy(pairwise_distances(X.T, X, metric=type)).float()
        if type == 'parcellating': #TODO: Does not seem to learn the structure.
            temp = ((self.X.unsqueeze(1) - self.X + 1e-06)**2).sum(-1)
            kernel = (2 * (temp - torch.diag(torch.diagonal(temp))))**0.5
        if type == 'normalised_x':
            kernel = self.X @ X.T / (self.X @ self.X.T).sum(0) #TODO: Sum row or column wise?
        if type == 'laplacian':
            D = torch.diag(self.X.sum(1))
            kernel = D - self.X #TODO: weird space..
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
