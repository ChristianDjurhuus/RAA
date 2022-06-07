import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import numpy as np
from collections import Counter
from torch_sparse import spspmm, transpose


# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing


class KAAsparse(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, k, data, data2, sample_size = 0.5, type = "sparse", seed_init = False,
                 data_type='sparse', non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None):
        super(KAAsparse, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_type = data_type
        self.sample_size = sample_size

        self.sparse_i_idx = data.to(self.device)
        self.sparse_j_idx = data2.to(self.device)
        self.non_sparse_i_idx_removed = non_sparse_i.to(self.device)
        self.non_sparse_j_idx_removed = non_sparse_j.to(self.device)
        self.sparse_i_idx_removed = sparse_i_rem.to(self.device)
        self.sparse_j_idx_removed = sparse_j_rem.to(self.device)
        self.removed_i = torch.cat((self.non_sparse_i_idx_removed, self.sparse_i_idx_removed))
        self.removed_j = torch.cat((self.non_sparse_j_idx_removed, self.sparse_j_idx_removed))
        self.N = int(self.sparse_j_idx.max() + 1)

        #self.edge_list = torch.stack((self.sparse_i_idx, self.sparse_j_idx), 0).to(self.device)
        temp1 = torch.stack((self.sparse_i_idx, self.sparse_j_idx), 0).to(self.device)
        temp2 = torch.stack((self.sparse_j_idx, self.sparse_i_idx), 0).to(self.device)
        self.edge_list = torch.cat((temp1,temp2),1)
        self.input_size = (self.N, self.N)
        self.k = k
        self.type = type.lower()
        self.K = self.kernel(self.edge_list)
        self.S = torch.nn.Parameter(torch.randn(self.k, self.N, device = self.device))
        self.C = torch.nn.Parameter(torch.randn(self.N, self.k, device = self.device))
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

    def kernel(self, edge_list):
        # check pairwise_distances
        degree_diag = torch.tensor(torch.unique(edge_list, return_counts=True)[1], dtype = torch.float32)
        # Create diag edge list
        diag_edgelist = torch.stack((torch.arange(degree_diag.shape[0], device = self.device), torch.arange(degree_diag.shape[0], device = self.device)), 0)
        degree_diag = torch.sparse_coo_tensor(diag_edgelist, degree_diag, (self.N, self.N), device = self.device)

        A = torch.sparse_coo_tensor(self.edge_list, torch.ones(self.edge_list.shape[1], device = self.device, dtype = torch.float32), (self.N, self.N), device=self.device)
        matrix_product = torch.sparse.mm((degree_diag**-1), A)
        index, value = transpose(matrix_product.coalesce().indices(), matrix_product.coalesce().values(), self.N, self.N)
        transposed_matrix_product = torch.sparse_coo_tensor(index, value, (self.N, self.N), device = self.device)

        kernel = torch.sparse.mm(matrix_product, transposed_matrix_product)
        x = torch.sparse_coo_tensor(kernel.coalesce().indices(), torch.ones(kernel.coalesce().values().shape, device = self.device), (self.N, self.N), device = self.device)
        kernel = x - kernel
        return kernel.float()

    def SSE(self):
        S = torch.softmax(self.S, dim=0)
        C = torch.softmax(self.C, dim=0)
        SSE = -2 * torch.trace(C.T @ torch.sparse.mm(self.K, S.T)) + torch.trace(C.T @ torch.sparse.mm(self.K, C) @ S @ S.T)
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