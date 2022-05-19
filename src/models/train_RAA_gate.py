from numpy import zeros
import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx 
import seaborn as sns
from torch_sparse import spspmm
import numpy as np

from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing

class RAA(nn.Module, Link_prediction, Preprocessing, Visualization):
    def __init__(self, A, edge_list, input_size, k, sample_size, sampling_weights, data_type='edge list', test_size=0.3, data_2=None):
        super(RAA, self).__init__()
        self.A = A
        self.edge_list = edge_list
        self.input_size = input_size
        self.k = k
        self.test_size = test_size
        self.data_type = data_type
        Preprocessing.__init__(self, data=edge_list, data_type=data_type, device=self.device, data_2=data_2)
        self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
        Link_prediction.__init__(self)
        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.softplus = nn.Softplus()
        self.a = torch.nn.Parameter(torch.randn(1))
        self.Z = torch.nn.Parameter(torch.randn(self.k, self.input_size[0]))
        #self.Z = torch.nn.Parameter(torch.load("src/models/S_initial.pt"))
        self.G = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))

        self.sampling_weights = sampling_weights
        self.sample_size = sample_size
        self.sparse_i_idx = edge_list[0]
        self.sparse_j_idx = edge_list[1]

    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx = torch.multinomial(self.sampling_weights, self.sample_size, replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator = torch.cat([sample_idx.unsqueeze(0), sample_idx.unsqueeze(0)], 0)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1]), indices_translator,
                                torch.ones(indices_translator.shape[1]), self.input_size[0], self.input_size[0],
                                self.input_size[0], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_translator, torch.ones(indices_translator.shape[1]), indexC, valueC,
                                self.input_size[0], self.input_size[0], self.input_size[0], coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_idx, sparse_i_sample, sparse_j_sample

    def log_likelihood(self):
        sample_idx, sparse_sample_i, sparse_sample_j = self.sample_network()

        Z = F.softmax(self.Z, dim=0)  # (K x N)
        G = F.sigmoid(self.G)  # Sigmoid activation function
        C = (Z.T * G) / (Z.T * G).sum(0)  # Gating function

        # For the nodes without links
        beta = self.beta[sample_idx].unsqueeze(1) + self.beta[sample_idx]  # (N x N)
        M = torch.mm(torch.mm(Z[:, sample_idx], C[sample_idx, :]), Z[:, sample_idx]).T
        mat = torch.exp(beta - self.softplus(self.a) * ((M.unsqueeze(1) - M + 1e-06) ** 2).sum(-1) ** 0.5)
        z_pdist1 = (0.5 * torch.mm(torch.exp(torch.ones(sample_idx.shape[0]).unsqueeze(0)),
                                   (torch.mm((mat - torch.diag(torch.diagonal(mat))),
                                             torch.exp(torch.ones(sample_idx.shape[0])).unsqueeze(-1)))))

        # For the nodes with links
        M = torch.mm(Z[:,sample_idx], C[sample_idx,:])  # This could perhaps be a computational issue
        beta = self.beta[sparse_sample_i] + self.beta[sparse_sample_j]
        z_pdist2 = (beta - self.softplus(self.a) * ((
            ((torch.mm(M, Z[:, sparse_sample_i]).T - torch.mm(M, Z[:, sparse_sample_j]).T + 1e-06) ** 2).sum(
                -1))) ** 0.5).sum()

        log_likelihood_sparse = z_pdist2 - z_pdist1
        return log_likelihood_sparse
    
