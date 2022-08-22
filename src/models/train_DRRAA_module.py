import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from torch_sparse import spspmm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing
from src.features.spectral_clustering import Spectral_clustering_init

class DRRAA(nn.Module, Preprocessing, Link_prediction, Visualization, Spectral_clustering_init):
    def __init__(self, k, d, sample_size, data, data_type = "edge list", data_2 = None, link_pred=False, test_size=0.3, non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None, seed_split = False, seed_init = False, init_Z = None, graph = False, values = None, missing_data=None):
        super(DRRAA, self).__init__()

        self.data_type = data_type
        self.link_pred = link_pred
        if seed_init != False:
            np.random.seed(seed_init)
            torch.manual_seed(seed_init)
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
            if non_sparse_i != None:
                self.non_sparse_i_idx_removed = non_sparse_i.to(self.device)
                self.non_sparse_j_idx_removed = non_sparse_j.to(self.device)
                self.sparse_i_idx_removed = sparse_i_rem.to(self.device)
                self.sparse_j_idx_removed = sparse_j_rem.to(self.device)
                self.removed_i = torch.cat((self.non_sparse_i_idx_removed, self.sparse_i_idx_removed))
                self.removed_j = torch.cat((self.non_sparse_j_idx_removed, self.sparse_j_idx_removed))
            
            self.N = int(self.sparse_j_idx.max() + 1)
            if values == None:
                self.values = torch.ones(self.sparse_i_idx.shape[1], device = self.device)
            else:
                self.values = values.float().to(self.device)
        
        Visualization.__init__(self)

        self.input_size = (self.N, self.N)
        self.k = k
        self.d = d

        ####Nakis contribution:
        self.missing_data = missing_data
        Spectral_clustering_init.__init__(self, num_of_eig=k, method="Adjacency")
        self.initialization=1
        self.scaling=1
        #self.flag1 ?
        self.pdist = nn.PairwiseDistance(p=2, eps=0)
        self.Softmax = nn.Softmax(1)

        self.spectral_data = self.spectral_clustering()
        spectral_centroids_to_z = self.spectral_data
        #N x K
        if self.spectral_data.shape[1] > self.k:
            self.latent_z1 = nn.Parameter(spectral_centroids_to_z[:,0:self.k])
        elif self.spectral_data.shape[1] == self.k:
            self.latent_z1 = nn.Parameter(spectral_centroids_to_z)
        else:
            self.latent_z1 = nn.Parameter(torch.zeros(self.N, self.k, device=self.device))
            self.latent_z1.data[:, 0:self.spectral_data.shape[1]]=spectral_centroids_to_z    
        ####

        self.beta = torch.nn.Parameter(torch.ones(self.input_size[0], device = self.device))
        self.softplus = nn.Softplus()
        self.A = torch.nn.Parameter(torch.randn(self.d, self.k, device = self.device))
        self.Gate = torch.nn.Parameter(torch.randn(self.input_size[0], self.k, device = self.device))

        self.missing_data = False
        self.sampling_weights = torch.ones(self.N, device = self.device)
        self.sample_size = round(sample_size * self.N) #TODO check if this need changing

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
            
       
        indexC, valueC = spspmm(edges, self.values.float(), indices_translator,
                                torch.ones(indices_translator.shape[1], device = self.device), self.input_size[0], self.input_size[0],
                                self.input_size[0], coalesced=True) #TODO ask if this is correct in terms of weighted networks

        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_translator, torch.ones(indices_translator.shape[1], device = self.device), indexC, valueC,
                                self.input_size[0], self.input_size[0], self.input_size[0], coalesced=True)


        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_idx, sparse_i_sample, sparse_j_sample, valueC

    def log_likelihood(self, epoch):

        self.epoch = epoch
        self.latent_z = self.Softmax(self.latent_z1)

        #Only optimize random effects
        if self.scaling:
            sample_idx, sparse_sample_i, sparse_sample_j, valueC = self.sample_network()
            mat = torch.exp(self.beta[sample_idx].unsqueeze(1) + self.beta[sample_idx])
            z_pdist1 = 0.5*(mat-torch.diag(torch.diagonal(mat))).sum()
            z_pdist2 = (self.beta[sparse_sample_i] + self.beta[sparse_sample_j]).sum()

            log_likelihood_sparse = z_pdist2 - z_pdist1

        # Full optimization
        else:

            self.latent_z = self.Softmax(self.latent_z1)
            self.G = torch.sigmoid(self.Gate)
            self.C = (self.latent_z * self.G) / (self.latent_z * self.G).sum(0) #Gating function
            #u, sigma, vt = torch.svd(self.A)
            #A = torch.diag(sigma)@vt.T
            AZC = self.A@(self.latent_z.transpose(0,1)@self.C)
            self.AZC = AZC
            sample_idx, sparse_sample_i, sparse_sample_j, valueC = self.sample_network()

            #For the nodes without links
            AZC_non_link = (AZC@(self.latent_z[sample_idx].transpose(0,1))).transpose(0,1)
            AZC_link_i = (AZC@(self.latent_z[sparse_sample_i].transpose(0,1))).transpose(0,1)
            AZC_link_j = (AZC@(self.latent_z[sparse_sample_j].transpose(0,1))).transpose(0,1)
            
            mat=torch.exp(-((torch.cdist(AZC_non_link, AZC_non_link, p=2))))
            z_pdist1 = 0.5*torch.mm(torch.exp(self.beta[sample_idx].unsqueeze(0)), (torch.mm((mat-torch.diag(torch.diagonal(mat))), torch.exp(self.beta[sample_idx].unsqueeze(-1)))))
            z_pdist2 = ((self.beta[sparse_sample_i] + self.beta[sparse_sample_j] - (((AZC_link_i-AZC_link_j+1e-06)**2).sum(-1)**0.5)) * valueC).sum()

            log_likelihood_sparse = z_pdist2 - z_pdist1

        return log_likelihood_sparse





    def train(self, iterations, LR = 0.01, print_loss = False):
        optimizer = torch.optim.Adam(params = self.parameters(), lr = LR)

        for epoch in range(iterations):
            if epoch == 500:
                self.scaling = 0
            loss = - self.log_likelihood(epoch=epoch) / self.sample_size #self.N
            self.losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if print_loss:
                print('Loss at the', epoch ,'epoch:',loss.item())

        