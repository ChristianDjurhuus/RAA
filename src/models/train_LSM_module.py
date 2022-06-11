from cmath import nan
import torch
import torch.nn as nn
from torch_sparse import spspmm
import numpy as np

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing

class LSM(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, d, sample_size, data, data_type = "Edge list", data_2 = None, link_pred=False, test_size = 0.3,
                 non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None,
                 seed_split = False, seed_init = False):
        super(LSM, self).__init__()
        self.data_type = data_type
        self.link_pred = link_pred
        if link_pred:
            self.test_size = test_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.data_type != "sparse":
            Preprocessing.__init__(self, data = data, data_type = data_type, device = self.device, data_2 = data_2)
            self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
            if link_pred:
                if seed_split != False:
                    np.random.seed(seed_split)
                    torch.manual_seed(seed_split)
                Link_prediction.__init__(self)
            if seed_init != False:
                np.random.seed(seed_init)
                torch.manual_seed(seed_init)
            Visualization.__init__(self)
            self.sparse_i_idx = self.edge_list[0]
            self.sparse_i_idx = self.sparse_i_idx.to(self.device)
            self.sparse_j_idx = self.edge_list[1]
            self.sparse_j_idx = self.sparse_j_idx.to(self.device)

        if self.data_type == "sparse":
            #create indices to index properly the receiver and senders variable
            if link_pred: 
                Link_prediction.__init__(self)
            self.sparse_i_idx = data.to(self.device)
            self.sparse_j_idx = data_2.to(self.device)
            self.non_sparse_i_idx_removed = non_sparse_i.to(self.device)
            self.non_sparse_j_idx_removed = non_sparse_j.to(self.device)
            self.sparse_i_idx_removed = sparse_i_rem.to(self.device)
            self.sparse_j_idx_removed = sparse_j_rem.to(self.device)
            self.removed_i = torch.cat((self.non_sparse_i_idx_removed, self.sparse_i_idx_removed))
            self.removed_j = torch.cat((self.non_sparse_j_idx_removed, self.sparse_j_idx_removed))
            self.N = int(self.sparse_j_idx.max() + 1)

        self.input_size = (self.N, self.N)
        self.latent_dim = d

        # initialize beta to follow a Uniform(3,5)
        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0], device=self.device))
        #self.beta = torch.nn.Parameter((3-5) * torch.rand(1, self.N, device = self.device) + 5)[0]

        self.latent_Z = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim, device = self.device))

        self.missing_data = False
        self.sampling_weights = torch.ones(self.N, device = self.device)
        self.sample_size = round(sample_size * self.N)

        # list for training loss
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

    def log_likelihood(self):
        sample_idx, sparse_sample_i, sparse_sample_j = self.sample_network()
        beta = self.beta[sample_idx].unsqueeze(1) + self.beta[sample_idx]  # (N x N)
        mat = torch.exp(beta-((self.latent_Z[sample_idx].unsqueeze(1) - self.latent_Z[sample_idx] + 1e-06) ** 2).sum(-1) ** 0.5)
        #For the nodes without links
        z_pdist1 = (0.5 * (mat - torch.diag(torch.diagonal(mat))).sum())
        #mat = ((self.latent_Z[sample_idx].unsqueeze(1) - self.latent_Z[sample_idx] + 1e-06) ** 2).sum(-1) ** 0.5
        #z_pdist1 = (0.5 * torch.exp((beta - (mat - torch.diag(torch.diagonal(mat))))).sum())



        #z_pdist1 = (0.5 * torch.mm(torch.exp(torch.ones(sample_idx.shape[0], device = self.device).unsqueeze(0)),
        #                                                  (torch.mm((mat - torch.diag(torch.diagonal(mat))),
        #                                                            torch.exp(torch.ones(sample_idx.shape[0], device = self.device)).unsqueeze(-1)))))

        #For the nodes with links
        z_pdist2 = (self.beta[sparse_sample_i]+self.beta[sparse_sample_j]-(((self.latent_Z[sparse_sample_i] - self.latent_Z[sparse_sample_j] + 1e-06) ** 2).sum(-1) ** 0.5) ).sum()

        log_likelihood_sparse = z_pdist2 - z_pdist1

        return log_likelihood_sparse


    def train(self, iterations, LR = 0.01, print_loss = False):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = - self.log_likelihood() / self.sample_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())


class LSMAA(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, d, k, sample_size, data, data_type = "Edge list", data_2 = None, link_pred=False, test_size = 0.3,
                 non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None,
                 seed_split = False, seed_init = False):
        super(LSMAA, self).__init__()
        self.data_type = data_type
        self.link_pred = link_pred
        if link_pred:
            self.test_size = test_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.data_type != "sparse":
            Preprocessing.__init__(self, data = data, data_type = data_type, device = self.device, data_2 = data_2)
            self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
            if link_pred:
                if seed_split != False:
                    np.random.seed(seed_split)
                    torch.manual_seed(seed_split)
                Link_prediction.__init__(self)
            if seed_init != False:
                np.random.seed(seed_init)
                torch.manual_seed(seed_init)
            Visualization.__init__(self)
            self.sparse_i_idx = self.edge_list[0]
            self.sparse_i_idx = self.sparse_i_idx.to(self.device)
            self.sparse_j_idx = self.edge_list[1]
            self.sparse_j_idx = self.sparse_j_idx.to(self.device)

        if self.data_type == "sparse":
            #create indices to index properly the receiver and senders variable
            if link_pred:
                Link_prediction.__init__(self)
            self.sparse_i_idx = data.to(self.device)
            self.sparse_j_idx = data_2.to(self.device)
            self.non_sparse_i_idx_removed = non_sparse_i.to(self.device)
            self.non_sparse_j_idx_removed = non_sparse_j.to(self.device)
            self.sparse_i_idx_removed = sparse_i_rem.to(self.device)
            self.sparse_j_idx_removed = sparse_j_rem.to(self.device)
            self.removed_i = torch.cat((self.non_sparse_i_idx_removed, self.sparse_i_idx_removed))
            self.removed_j = torch.cat((self.non_sparse_j_idx_removed, self.sparse_j_idx_removed))
            self.N = int(self.sparse_j_idx.max() + 1)

        self.input_size = (self.N, self.N)
        self.latent_dim = d
        self.k = k

        # initialize beta to follow a Uniform(3,5)
        self.beta = torch.nn.Parameter(torch.randn(1, device=self.device))
        #self.beta = torch.nn.Parameter((3-5) * torch.rand(1, self.N, device = self.device) + 5)[0]

        self.latent_Z = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim, device = self.device))

        self.missing_data = False
        self.sampling_weights = torch.ones(self.N, device = self.device)
        self.sample_size = round(sample_size * self.N)

        # list for training loss
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

    def log_likelihood(self):
        sample_idx, sparse_sample_i, sparse_sample_j = self.sample_network()
        beta = self.beta[sample_idx].unsqueeze(1) + self.beta[sample_idx] #(N x N)
        mat = torch.exp(beta-((self.latent_Z[sample_idx].unsqueeze(1) - self.latent_Z[sample_idx] + 1e-06) ** 2).sum(-1) ** 0.5)
        #For the nodes without links
        z_pdist1 = (0.5 * torch.mm(torch.exp(torch.ones(sample_idx.shape[0], device = self.device).unsqueeze(0)),
                                                          (torch.mm((mat - torch.diag(torch.diagonal(mat))),
                                                                    torch.exp(torch.ones(sample_idx.shape[0], device = self.device)).unsqueeze(-1)))))

        #For the nodes with links
        z_pdist2 = ((self.beta[sparse_sample_i] + self.beta[sparse_sample_j]) - (((self.latent_Z[sparse_sample_i] - self.latent_Z[sparse_sample_j] + 1e-06) ** 2).sum(-1) ** 0.5) ).sum()

        log_likelihood_sparse = z_pdist2 - z_pdist1

        return log_likelihood_sparse


    def train(self, iterations, LR = 0.01, print_loss = False):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = - self.log_likelihood() / self.sample_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())
