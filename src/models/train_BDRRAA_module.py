import torch
import torch.nn as nn
from torch_sparse import spspmm


from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction

class BDRRAA(nn.Module, Link_prediction, Visualization):
    def __init__(self, k, d, sample_size, data, data_type = "sparse", data2 = None, non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None, missing=None):
        super(BDRRAA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_type = data_type
        self.sample_size = sample_size
        self.k = k
        self.d = d

        self.sparse_i_idx = data.to(self.device)
        self.sparse_j_idx = data2.to(self.device)
        self.non_sparse_i_idx_removed = non_sparse_i.to(self.device)
        self.non_sparse_j_idx_removed = non_sparse_j.to(self.device)
        self.sparse_i_idx_removed = sparse_i_rem.to(self.device)
        self.sparse_j_idx_removed = sparse_j_rem.to(self.device)
        self.removed_i = torch.cat((self.non_sparse_i_idx_removed, self.sparse_i_idx_removed),dim=0)
        self.removed_j = torch.cat((self.non_sparse_j_idx_removed, self.sparse_j_idx_removed),dim=0)

        self.sample_shape = (len(self.sparse_i_idx), len(self.sparse_j_idx))
        self.sampling_i_weights = torch.ones(self.sample_shape[0], device = self.device)
        self.sampling_j_weights = torch.ones(self.sample_shape[1], device = self.device)

        self.sample_i_size = int(self.sample_shape[0] * self.sample_size)
        self.sample_j_size = int(self.sample_shape[1] * self.sample_size)

        self.beta = torch.nn.Parameter(torch.randn(self.sample_shape[0], device = self.device))
        self.gamma = torch.nn.Parameter(torch.randn(self.sample_shape[1], device = self.device))
        self.softplus = nn.Softplus()
        self.A = torch.nn.Parameter(torch.randn(self.d, self.k, device = self.device))

        self.Z_i = torch.nn.Parameter(torch.randn(self.k, self.sample_shape[0], device = self.device))
        self.Z_j = torch.nn.Parameter(torch.randn(self.k, self.sample_shape[1], device = self.device))

        self.Gate = torch.nn.Parameter(torch.randn(self.sample_shape[0] + self.sample_shape[1], self.k, device = self.device))

        self.losses = []
        self.N = self.sample_shape[0] + self.sample_shape[1]
        self.missing = missing
        Link_prediction.__init__(self)
        Visualization.__init__(self)

    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_i_idx = torch.multinomial(self.sampling_i_weights, self.sample_i_size, replacement=False)
        sample_j_idx = torch.multinomial(self.sampling_j_weights, self.sample_j_size, replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_i_translator = torch.cat([sample_i_idx.unsqueeze(0), sample_i_idx.unsqueeze(0)], 0)
        indices_j_translator = torch.cat([sample_j_idx.unsqueeze(0), sample_j_idx.unsqueeze(0)], 0)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1], device=self.device), indices_j_translator,
                                                        torch.ones(indices_j_translator.shape[1], device = self.device), self.sample_shape[0], self.sample_shape[1],
                                                        self.sample_shape[1], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1], device = self.device), indexC, valueC,
                                self.sample_shape[0], self.sample_shape[0], self.sample_shape[1], coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_i_idx, sample_j_idx, sparse_i_sample, sparse_j_sample

    def log_likelihood(self):
        sample_i_idx, sample_j_idx, sparse_sample_i, sparse_sample_j = self.sample_network()
        Z_i = torch.softmax(self.Z_i, dim=0)  # (K x N)
        Z_j = torch.softmax(self.Z_j, dim=0)
        Z = torch.cat((Z_i, Z_j),1) #Concatenate partition embeddings
        Gate = torch.cat((self.Gate, self.Gate), 0)
        Gate = torch.sigmoid(Gate)  # Sigmoid activation function
        C = (Z.T * Gate) / (Z.T * Gate).sum(0)  # Gating function
        # For the nodes without links
        bias_matrix = self.beta[sample_i_idx].unsqueeze(1) + self.gamma[sample_j_idx]  # (N x N)
        AZC = torch.mm(self.A, torch.mm(Z, C))
        mat = (torch.exp(bias_matrix -
                         ((torch.mm(AZC,Z_i[:,sample_i_idx]).T.unsqueeze(1) -
                           torch.mm(AZC,Z_j[:,sample_j_idx]).T + 1e-06) ** 2).sum(-1) ** 0.5)).sum()
        if self.missing!=None:
            mat_missing = (torch.exp(bias_matrix-((torch.mm(AZC,Z_i[:,self.missing[:,0]]).T.unsqueeze(1) -
                           torch.mm(AZC,Z_j[:,self.missing[:,1]]).T + 1e-06) ** 2).sum(-1) ** 0.5)).sum()


        #nodes with links
        mat_links = ((self.beta[sparse_sample_i] + self.gamma[sparse_sample_j]) -
                     (((AZC @ Z_i[:,sparse_sample_i]).T -
                       (AZC @ Z_j[:,sparse_sample_j]).T + 1e-06) ** 2).sum(-1) ** 0.5).sum()
        log_likelihood_sparse = mat_links - mat
        return log_likelihood_sparse

    def train(self, iterations, LR = 0.1, print_loss = False):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = - self.log_likelihood() / self.N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())