import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
import networkx as nx 
import numpy as np
#import umap
#import umap.plot
from torch_sparse import spspmm
import archetypes

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing

class LSM(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, latent_dim, sample_size, data, data_type = "Edge list", data_2 = None):
        super(LSM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Preprocessing.__init__(self, data = data, data_type = data_type, device = self.device, data_2 = data_2)
        self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
        Link_prediction.__init__(self)

        Visualization.__init__(self)

        self.input_size = (self.N, self.N)
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn((self.N), device = self.device))
        self.latent_Z = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim, device = self.device))

        self.missing_data = False
        self.sampling_weights = torch.ones(self.N, device = self.device)
        self.sample_size = round(sample_size * self.N)
        self.sparse_i_idx = self.edge_list[0]
        self.sparse_i_idx = self.sparse_i_idx.to(self.device)
        self.sparse_j_idx = self.edge_list[1]
        self.sparse_j_idx = self.sparse_j_idx.to(self.device)
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
        z_pdist1 = (0.5 * torch.mm(torch.exp(torch.ones(sample_idx.shape[0], device = self.device).unsqueeze(0)),
                                                          (torch.mm((mat - torch.diag(torch.diagonal(mat))),
                                                                    torch.exp(torch.ones(sample_idx.shape[0], device = self.device)).unsqueeze(-1)))))

        #For the nodes with links
        z_pdist2 = (self.beta[sparse_sample_i]+self.beta[sparse_sample_j]-(((self.latent_Z[sparse_sample_i] - self.latent_Z[sparse_sample_j] + 1e-06) ** 2).sum(-1) ** 0.5) ).sum()

        log_likelihood_sparse = z_pdist2 - z_pdist1

        return log_likelihood_sparse


    def train(self, iterations, LR = 0.01, print_loss = True):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = - self.log_likelihood() / self.N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())


class LSMAA(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, latent_dim,k, sample_size, data, data_type = "Edge list", data_2 = None):
        super(LSMAA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Preprocessing.__init__(self, data=data, data_type=data_type, device=self.device, data_2=data_2)
        self.edge_list, self.N, self.G = Preprocessing.convert_to_egde_list(self)
        Link_prediction.__init__(self)
        Visualization.__init__(self)

        self.input_size = (self.N, self.N)
        self.latent_dim = latent_dim
        self.latent_Z = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim, device = self.device))
        self.k = k

        self.beta = torch.nn.Parameter(torch.randn(1, device = self.device))



        self.missing_data = False
        self.sampling_weights = torch.ones(self.N, device = self.device)
        self.sample_size = round(sample_size * self.N)
        self.sparse_i_idx = self.edge_list[0]
        self.sparse_i_idx = self.sparse_i_idx.to(self.device)
        self.sparse_j_idx = self.edge_list[1]
        self.sparse_j_idx = self.sparse_j_idx.to(self.device)
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
        mat = torch.exp(self.beta-((self.latent_Z[sample_idx].unsqueeze(1) - self.latent_Z[sample_idx] + 1e-06) ** 2).sum(-1) ** 0.5)
        #For the nodes without links
        z_pdist1 = (0.5 * torch.mm(torch.exp(torch.ones(sample_idx.shape[0], device = self.device).unsqueeze(0)),
                                                          (torch.mm((mat - torch.diag(torch.diagonal(mat))),
                                                                    torch.exp(torch.ones(sample_idx.shape[0], device = self.device)).unsqueeze(-1)))))

        #For the nodes with links
        z_pdist2 = (self.beta-(((self.latent_Z[sparse_sample_i] - self.latent_Z[sparse_sample_j] + 1e-06) ** 2).sum(-1) ** 0.5) ).sum()

        log_likelihood_sparse = z_pdist2 - z_pdist1

        return log_likelihood_sparse


    def train(self, iterations, LR = 0.01, print_loss = True):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = - self.log_likelihood() / self.N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())
if __name__ == "__main__": 
    seed = 1984
    torch.random.manual_seed(seed)


    ZKC_graph = nx.karate_club_graph()
    #Let's keep track of which nodes represent John A and Mr Hi
    Mr_Hi = 0
    John_A = 33

    #Get the edge list
    edge_list = np.array(list(map(list,ZKC_graph.edges()))).T
    #edge_list.sort(axis=1)  #TODO: Sort in order to recieve the upper triangular part of the adjacency matrix
    edge_list = torch.from_numpy(edge_list).long()

    #Get N and latent_dim (k)
    N = len(ZKC_graph.nodes())
    latent_dim = 2

    #Let's display the labels of which club each member ended up joining
    club_labels = nx.get_node_attributes(ZKC_graph,'club')

    #Getting adjacency matrix
    A = nx.convert_matrix.to_numpy_matrix(ZKC_graph)
    A = torch.from_numpy(A)
    AA = True
    link_pred = True

    if link_pred:
        num_samples = 15
        idx_i_test = torch.multinomial(input=torch.arange(0, float(N)), num_samples=num_samples,
                                       replacement=True)
        idx_j_test = torch.tensor(np.zeros(num_samples)).long()
        for i in range(len(idx_i_test)):
            idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(N))[
                torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(N)), num_samples=1,
                                  replacement=True).item()].item()  # Temp solution to sample from upper corner

        test = torch.stack((idx_i_test,idx_j_test))

        #TODO: could be a killer.. maybe do it once and save adjacency list ;)
        def if_edge(a, edge_list):
            a = a.tolist()
            edge_list = edge_list.tolist()
            a = list(zip(a[0], a[1]))
            edge_list = list(zip(edge_list[0], edge_list[1]))
            return [a[i] in edge_list for i in range(len(a))]

        target = if_edge(test, edge_list)


    model = LSM(input_size = (N,N), latent_dim=latent_dim, sampling_weights=torch.ones(N), sample_size=34, edge_list=edge_list)
    optimizer = torch.optim.Adam(params=model.parameters())
    
    losses = []
    iterations = 1000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the',_,'iteration:',loss.item())



    
    #Link prediction
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(target, idx_i_test, idx_j_test)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
        plt.plot([0, 1], [0, 1],'r--', label='random')
        plt.legend(loc = 'lower right')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Latent space model")
        plt.show()


    labels = list(club_labels.values())
    idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    latent_Z = model.latent_Z.detach().numpy()

    if latent_Z.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(latent_Z[:,0][idx_hi], latent_Z[:,1][idx_hi], latent_Z[:,2][idx_hi], c = 'red', label='Mr. Hi' )
        ax.scatter(latent_Z[:,0][idx_of], latent_Z[:,1][idx_of], latent_Z[:,2][idx_of], c = 'blue', label='Officer')
        ax.text(latent_Z[Mr_Hi,0], latent_Z[Mr_Hi,1], latent_Z[Mr_Hi,2], 'Mr. Hi')
        ax.text(latent_Z[John_A, 0], latent_Z[John_A, 1], latent_Z[John_A, 2],  'Officer')
        ax.set_title(f"Latent space after {iterations} iterations")
        ax.legend()
        plt.show()

    if latent_Z.shape[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(latent_Z[:,0][idx_hi], latent_Z[:,1][idx_hi], c = 'red', label='Mr. Hi')
        ax1.scatter(latent_Z[:,0][idx_of], latent_Z[:,1][idx_of], c = 'blue', label='Officer')
        ax1.annotate('Mr. Hi', latent_Z[Mr_Hi,:])
        ax1.annotate('Officer', latent_Z[John_A, :])
        ax1.legend()
        ax1.set_title(f"Latent space after {iterations} iterations")
        #Plotting learning curve
        ax2.plot(losses)
        ax2.set_title("Loss")
        plt.show()


    if AA:
        import archetypes
        aa = archetypes.AA(n_archetypes=2)
        latent_Z_trans = aa.fit_transform(model.latent_Z.detach().numpy())

        labels = list(club_labels.values())
        idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
        idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

        if latent_Z_trans.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(latent_Z_trans[:, 0][idx_hi], latent_Z_trans[:, 1][idx_hi], latent_Z_trans[:, 2][idx_hi], c='red', label='Mr. Hi')
            ax.scatter(latent_Z_trans[:, 0][idx_of], latent_Z_trans[:, 1][idx_of], latent_Z_trans[:, 2][idx_of], c='blue',
                       label='Officer')
            ax.text(latent_Z_trans[Mr_Hi, 0], latent_Z_trans[Mr_Hi, 1], latent_Z_trans[Mr_Hi, 2], 'Mr. Hi')
            ax.text(latent_Z_trans[John_A, 0], latent_Z_trans[John_A, 1], latent_Z_trans[John_A, 2], 'Officer')
            ax.set_title(f"Latent space + AA after {iterations} iterations")
            ax.legend()
            plt.show()

        if latent_Z_trans.shape[1] == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.scatter(latent_Z_trans[:, 0][idx_hi], latent_Z_trans[:, 1][idx_hi], c='red', label='Mr. Hi')
            ax1.scatter(latent_Z_trans[:, 0][idx_of], latent_Z_trans[:, 1][idx_of], c='blue', label='Officer')
            ax1.annotate('Mr. Hi', latent_Z_trans[Mr_Hi, :])
            ax1.annotate('Officer', latent_Z_trans[John_A, :])
            ax1.legend()
            ax1.set_title(f"Latent space + AA after {iterations} iterations")
            # Plotting learning curve
            ax2.plot(losses)
            ax2.set_title("Loss")
        plt.show()
