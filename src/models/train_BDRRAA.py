from numpy import zeros
import torch
import torch.nn as nn
from scipy.io import mmread
import scipy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx
import seaborn as sns
from torch_sparse import spspmm
import numpy as np


class BDRRAA(nn.Module):
    def __init__(self, input_size: tuple, k: int, d: int, sampling_weights: tuple, sample_size: tuple, edge_list):
    #def __init__(self, k, d, sample_size, data, data_type = "edge list", data_2 = None, link_pred=False, test_size=0.3, non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None, seed_split = False, seed_init = False):
        super(BDRRAA, self).__init__()
        self.input_size = input_size
        self.k = k
        self.d = d

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))
        self.softplus = nn.Softplus()
        self.A = torch.nn.Parameter(torch.randn(self.d, self.k))

        self.Z_i = torch.nn.Parameter(torch.randn(self.k, self.input_size[0]))
        self.Z_j = torch.nn.Parameter(torch.randn(self.k, self.input_size[1]))

        self.G = torch.nn.Parameter(torch.randn(self.input_size[0]+self.input_size[1], self.k))

        self.sampling_i_weights = sampling_weights[0]
        self.sampling_j_weights = sampling_weights[1]
        self.sample_i_size = sample_size[0]
        self.sample_j_size = sample_size[1]
        self.sparse_i_idx = edge_list[0]
        self.sparse_j_idx = edge_list[1]

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
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1]), indices_j_translator,
                                torch.ones(indices_j_translator.shape[1]), self.input_size[0], self.input_size[1],
                                self.input_size[1], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1]), indexC, valueC,
                                self.input_size[0], self.input_size[0], self.input_size[1], coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_i_idx, sample_j_idx, sparse_i_sample, sparse_j_sample

    def log_likelihood(self):
        sample_i_idx, sample_j_idx, sparse_sample_i, sparse_sample_j = self.sample_network()
        Z_i = F.softmax(self.Z_i, dim=0)  # (K x N)
        Z_j = F.softmax(self.Z_j, dim=0)
        Z = torch.cat((Z_i[:,sample_i_idx], Z_j[:,sample_j_idx]),1) #Concatenate partition embeddings
        G = torch.cat((self.G[sample_i_idx,:], self.G[sample_j_idx,:]), 0)
        G = torch.sigmoid(G)  # Sigmoid activation function
        C = (Z.T * G) / (Z.T * G).sum(0)  # Gating function
        # For the nodes without links
        bias_matrix = self.beta[sample_i_idx].unsqueeze(1) + self.gamma[sample_j_idx]  # (N x N)
        AZC = torch.mm(self.A, torch.mm(Z, C))
        mat = (torch.exp(bias_matrix -
                         ((torch.mm(AZC,Z_i[:,sample_i_idx]).T.unsqueeze(1) -
                           torch.mm(AZC,Z_j[:,sample_j_idx]).T + 1e-06) ** 2).sum(-1) ** 0.5)).sum()
        mat_links = ((self.beta[sparse_sample_i] + self.gamma[sparse_sample_j]) -
                     (((AZC @ Z_i[:,sparse_sample_i]).T -
                       (AZC @ Z_j[:,sparse_sample_j]).T + 1e-06) **2).sum(-1)).sum()
        log_likelihood_sparse = mat_links - mat
        return log_likelihood_sparse



    def link_prediction(self, X_test, idx_i_test, idx_j_test):
        with torch.no_grad():
            Z_i = F.softmax(self.Z_i, dim=0)  # (K x N)
            Z_j = F.softmax(self.Z_j, dim=0)
            Z = torch.cat((Z_i[:,idx_i_test], Z_j[:,idx_j_test]),1) #Concatenate partition embeddings
            G = torch.cat((self.G[idx_i_test,:], self.G[idx_j_test,:]), 0)

            Z = F.softmax(Z, dim=0)
            G = F.sigmoid(G)
            C = (Z.T * G) / (Z.T * G).sum(0)  # Gating function

            M_i = torch.matmul(self.A,
                               torch.matmul(torch.matmul(Z, C), Z[:, idx_i_test])).T  # Size of test set e.g. K x N
            M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, idx_j_test])).T
            z_pdist_test = ((M_i - M_j + 1e-06) ** 2).sum(-1) ** 0.5  # N x N
            theta = (self.beta[idx_i_test] + self.gamma[idx_j_test] - z_pdist_test)  # N x N

            # Get the rate -> exp(log_odds)
            rate = torch.exp(theta)  # N

            fpr, tpr, threshold = metrics.roc_curve(target, rate.numpy())

            # Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target, rate.cpu().data.numpy())

            return auc_score, fpr, tpr


if __name__ == "__main__":
    from scipy.io import mmread
    import networkx as nx
    import igraph

    #g = igraph.read("data/toy_data/divorce/divorce.mtx", format="edge")
    G = mmread('data/toy_data/divorce/divorce.mtx')
    #https://stackoverflow.com/questions/49222857/modulenotfounderror-no-module-named-graph-tool
    import graph_tools as gt
    #g = gt.collection.ns["board_directors/net2m_2002-05-01"]
    #G = nx.read_graphml("data/raw/crime/network.xml")
    edge_list = torch.tensor([G.row,G.col]).T
    edge_list = edge_list.long()
    seed = 4
    torch.random.manual_seed(seed)

    # A = mmread("data/raw/soc-karate.mtx")
    # A = A.todense()
    k = 3
    d = 2

    link_pred = True

    if link_pred:
        num_samples = round(0.3 * ((50 * 9)))
        idx_i_test = torch.multinomial(input=torch.arange(0, float(50)), num_samples=num_samples,
                                       replacement=True)
        idx_j_test = torch.multinomial(input=torch.arange(0, float(9)), num_samples=num_samples, replacement=True)

        test = torch.stack((idx_i_test, idx_j_test))

        # TODO: could be a killer.. maybe do it once and save adjacency list ;)
        def if_edge(a, edge_list):
            a = a.tolist()
            edge_list = edge_list.tolist()
            a = list(zip(a[0], a[1]))
            edge_list = list(zip(edge_list[0], edge_list[1]))
            return [a[i] in edge_list for i in range(len(a))]

        target = [] #if_edge(test, edge_list)
        G = G.todense()
        for i in range(len(idx_i_test)):
            if G[idx_i_test[i], idx_j_test[i]] == 1:
                G[idx_i_test[i], idx_j_test[i]] = 0
                target.append(True)
            else:
                target.append(False)

        G = scipy.sparse.coo_matrix(G)
        edge_list = torch.tensor([G.row,G.col]).T
        edge_list = edge_list.long()



    model = BDRRAA(input_size=(50, 9), k=k, d = d,sampling_weights=(torch.ones(50),torch.ones(9)), sample_size=(20,5), edge_list=edge_list)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.05)

    losses = []
    iterations = 20000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the', _, 'iteration:', loss.item())

    # Link prediction
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(target, idx_i_test, idx_j_test)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc_score)
        plt.plot([0, 1], [0, 1], 'r--', label='random')
        plt.legend(loc='lower right')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("RAA model")
        plt.show()

    # Plotting latent space
    Z_i = F.softmax(model.Z_i, dim=0)
    Z_j = F.softmax(model.Z_j, dim=0)
    Z = torch.cat((Z_i,Z_j),1)
    G = torch.sigmoid(model.G)
    C = (Z.T * G) / (Z.T * G).sum(0)


    embeddings = torch.matmul(model.A, torch.matmul(torch.matmul(Z, C), Z)).T
    #embeddings_j = torch.matmul(model.A_j, torch.matmul(torch.matmul(Z_j, C_j), Z_j)).T
    archetypes = torch.matmul(model.A, torch.matmul(Z, C))
    #archetypes_j = torch.matmul(model.A_j, torch.matmul(Z_j, C_j))

    # labels = list(club_labels.values())
    # idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    # idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2)
    sns.heatmap(Z.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax1)
    sns.heatmap(C.T.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax2)
    #sns.heatmap(Z_j.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax3)
    #sns.heatmap(C_j.T.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax4)

    if embeddings.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
                   embeddings[:, 2].detach().numpy(), c='red')
        ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
                   archetypes[2, :].detach().numpy(), marker='^', c='black')
        '''ax.scatter(embeddings_j[:, 0].detach().numpy(), embeddings_j[:, 1].detach().numpy(),
                   embeddings_j[:, 2].detach().numpy(), c='blue')
        ax.scatter(archetypes_j[0, :].detach().numpy(), archetypes_j[1, :].detach().numpy(),
                   archetypes_j[2, :].detach().numpy(), marker='^', c='purple')'''
        ax.set_title(f"Latent space after {iterations} iterations")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(embeddings[model.input_size[0]:, 0].detach().numpy(), embeddings[model.input_size[0]:, 1].detach().numpy(), c='red')
        ax1.scatter(embeddings[:model.input_size[0], 0].detach().numpy(), embeddings[:model.input_size[0], 1].detach().numpy(), c='blue')
        ax1.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(), marker='^', c='black')
        #ax1.scatter(embeddings_j[:, 0].detach().numpy(), embeddings_j[:, 1].detach().numpy(), c='blue')
        #ax1.scatter(archetypes_j[0, :].detach().numpy(), archetypes_j[1, :].detach().numpy(), marker='^', c='purple')
        ax1.set_title(f"Latent space after {iterations} iterations")
        # Plotting learning curve
        ax2.plot(losses)
        ax2.set_title("Loss")
    plt.show()