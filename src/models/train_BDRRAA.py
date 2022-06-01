import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import seaborn as sns
from torch_sparse import spspmm
import numpy as np

from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction

class BDRRAA(nn.Module, Link_prediction, Visualization):
    def __init__(self, k, d, sample_size, data, data_type = "sparse", data2 = None, non_sparse_i = None, non_sparse_j = None, sparse_i_rem = None, sparse_j_rem = None):
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
        self.removed_i = torch.cat((self.non_sparse_i_idx_removed, self.sparse_i_idx_removed))
        self.removed_j = torch.cat((self.non_sparse_j_idx_removed, self.sparse_j_idx_removed))

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
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1]), indices_j_translator,
                                torch.ones(indices_j_translator.shape[1]), self.sample_shape[0], self.sample_shape[1],
                                self.sample_shape[1], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1]), indexC, valueC,
                                self.sample_shape[0], self.sample_shape[0], self.sample_shape[1], coalesced=True)

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
        Gate = torch.cat((self.Gate[sample_i_idx,:], self.Gate[sample_j_idx,:]), 0)
        Gate = torch.sigmoid(Gate)  # Sigmoid activation function
        C = (Z.T * Gate) / (Z.T * Gate).sum(0)  # Gating function
        # For the nodes without links
        bias_matrix = self.beta[sample_i_idx].unsqueeze(1) + self.gamma[sample_j_idx]  # (N x N)
        AZC = torch.mm(self.A, torch.mm(Z, C))
        mat = (torch.exp(bias_matrix -
                         ((torch.mm(AZC,Z_i[:,sample_i_idx]).T.unsqueeze(1) -
                           torch.mm(AZC,Z_j[:,sample_j_idx]).T + 1e-06) ** 2).sum(-1) ** 0.5)).sum()
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


if __name__ == "__main__":
    seed = 42
    torch.random.manual_seed(seed)
    k = 3
    d = 2

    # Data
    dataset = "drug-gene"
    data = torch.from_numpy(np.loadtxt("../data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("../data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("../data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("../data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("../data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("../data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    model = BDRRAA(k = k, d = d, sample_size = 0.2, data = data, data2 = data2, non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem)
    iterations = 10000
    model.train(interations = iterations, print_loss = True)
    # Plotting latent space
    Z_i = F.softmax(model.Z_i, dim=0)
    Z_j = F.softmax(model.Z_j, dim=0)
    Z = torch.cat((Z_i,Z_j),1)
    G = torch.sigmoid(model.Gate)
    C = (Z.T * G) / (Z.T * G).sum(0)

    embeddings = torch.matmul(model.A, torch.matmul(torch.matmul(Z, C), Z)).T
    archetypes = torch.matmul(model.A, torch.matmul(Z, C))


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
        ax2.plot(model.losses)
        ax2.set_yscale("log")
        ax2.set_title("Loss")
    plt.show()

    model.plot_auc()




"""
    def link_prediction(self):
        with torch.no_grad():
            Z_i = F.softmax(self.Z_i, dim=0)  # (K x N)
            Z_j = F.softmax(self.Z_j, dim=0)
            Z = torch.cat((Z_i, Z_j),1) #Concatenate partition embeddings
            #Z = F.softmax(Z, dim=0)
            G = F.sigmoid(self.G)
            C = (Z.T * G) / (Z.T * G).sum(0)  # Gating function
            
                
            M_i = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.removed_i])).T  # Size of test set e.g. K x N
            M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.removed_j])).T
            z_pdist_test = ((M_i - M_j + 1e-06) ** 2).sum(-1) ** 0.5  # N x N
            theta = (self.beta[self.removed_i] + self.gamma[self.removed_j] - z_pdist_test)  # N x N

            # Get the rate -> exp(log_odds)
            rate = torch.exp(theta)  # N

            # TODO Skal lige ha tjekket om det er den rigtige rækkefølge.
            target = torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]), torch.ones(self.sparse_i_idx_removed.shape[0])))

            fpr, tpr, threshold = metrics.roc_curve(target, rate.numpy())

            # Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target, rate.cpu().data.numpy())

            return auc_score, fpr, tpr



    G = mmread('data/toy_data/divorce/divorce.mtx')

    edge_list = torch.tensor([G.row,G.col]).T
    edge_list = edge_list.long()
    seed = 42
    torch.random.manual_seed(seed)

    # A = mmread("data/raw/soc-karate.mtx")
    # A = A.todense()
    k = 3
    d = 2

    link_pred = True

    if link_pred:
        num_samples = round(0.2 * ((50 * 9)))
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

"""