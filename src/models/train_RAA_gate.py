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

class RAA(nn.Module):
    def __init__(self, A, edge_list, input_size, k, sample_size, sampling_weights):
        super(RAA, self).__init__()
        self.A = A
        self.edge_list = edge_list
        self.input_size = input_size
        self.k = k

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
    
    def link_prediction(self, A_test, idx_i_test, idx_j_test):
        with torch.no_grad():
            Z = F.softmax(self.Z, dim=0)
            G = F.sigmoid(self.G)
            C = (Z.T * G) / (Z.T * G).sum(0) #Gating function

            M_i = torch.matmul(torch.matmul(Z, C), Z[:, idx_i_test]).T #Size of test set e.g. K x N
            M_j = torch.matmul(torch.matmul(Z, C), Z[:, idx_j_test]).T
            z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
            theta = (self.beta[idx_i_test] + self.beta[idx_j_test] - self.a * z_pdist_test) # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            #Create target (make sure its in the right order by indexing)
            target = A_test[idx_i_test, idx_j_test] #N


            fpr, tpr, threshold = metrics.roc_curve(target.numpy(), rate.numpy())


            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())

            return auc_score, fpr, tpr


if __name__ == "__main__": 
    seed = 4
    torch.random.manual_seed(seed)

    #A = mmread("data/raw/soc-karate.mtx")
    #A = A.todense()
    ZKC_graph = nx.karate_club_graph()
    #Let's keep track of which nodes represent John A and Mr Hi
    Mr_Hi = 0
    John_A = 33

    # Get the edge list
    edge_list = np.array(list(map(list, ZKC_graph.edges()))).T
    # edge_list.sort(axis=1)  #TODO: Sort in order to recieve the upper triangular part of the adjacency matrix
    edge_list = torch.from_numpy(edge_list).long()

    # Get N and latent_dim (k)
    N = len(ZKC_graph.nodes())
    k = 2

    #Let's display the labels of which club each member ended up joining
    club_labels = nx.get_node_attributes(ZKC_graph,'club')

    #Getting adjacency matrix
    A = nx.convert_matrix.to_numpy_matrix(ZKC_graph)
    A = torch.from_numpy(A)


    link_pred = True

    if link_pred:
        A_shape = A.shape
        num_samples = 15
        idx_i_test = torch.multinomial(input=torch.arange(0,float(A_shape[0])), num_samples=num_samples,
                                       replacement=True)
        idx_j_test = torch.tensor(zeros(num_samples)).long()
        for i in range(len(idx_i_test)):
            idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(A_shape[1]))[torch.multinomial(input = torch.arange(idx_i_test[i].item(), float(A_shape[1])), num_samples=1, replacement=True).item()].item() #Temp solution to sample from upper corner
        
        #idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
        #                               replacement=True)
        
        A_test = A.detach().clone()
        A_test[:] = 0
        A_test[idx_i_test, idx_j_test] = A[idx_i_test,idx_j_test]
        A[idx_i_test, idx_j_test] = 0

    model = RAA(A=A,input_size = (N,N),k=k, sampling_weights=torch.ones(N), sample_size=20, edge_list=edge_list)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    
    losses = []
    iterations = 10000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the',_,'iteration:',loss.item())
    
    #Link prediction
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(A_test, idx_i_test, idx_j_test)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
        plt.plot([0, 1], [0, 1],'r--', label='random')
        plt.legend(loc = 'lower right')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("RAA model")
        plt.show()

    #Plotting latent space
    Z = F.softmax(model.Z, dim=0)
    G = F.sigmoid(model.G)
    C = (Z.T * G) / (Z.T * G).sum(0)

    embeddings = torch.matmul(torch.matmul(Z, C), Z).T
    archetypes = torch.matmul(Z, C)

    labels = list(club_labels.values())
    idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(Z.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax1)
    sns.heatmap(C.T.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax2)

    if embeddings.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(embeddings[:,0].detach().numpy()[idx_hi], embeddings[:,1].detach().numpy()[idx_hi], embeddings[:,2].detach().numpy()[idx_hi], c = 'red', label='Mr. Hi' )
        ax.scatter(embeddings[:,0].detach().numpy()[idx_of], embeddings[:,1].detach().numpy()[idx_of], embeddings[:,2][idx_of].detach().numpy(), c = 'blue', label='Officer')
        ax.scatter(archetypes[0,:].detach().numpy(), archetypes[1,:].detach().numpy(), archetypes[2,:].detach().numpy(), marker = '^', c='black')
        ax.text(embeddings[Mr_Hi,0].detach().numpy(), embeddings[Mr_Hi,1].detach().numpy(), embeddings[Mr_Hi,2].detach().numpy(), 'Mr. Hi')
        ax.text(embeddings[John_A, 0].detach().numpy(), embeddings[John_A, 1].detach().numpy(), embeddings[John_A, 2].detach().numpy(),  'Officer')
        ax.set_title(f"Latent space after {iterations} iterations")
        ax.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(embeddings[:,0].detach().numpy()[idx_hi], embeddings[:,1].detach().numpy()[idx_hi], c = 'red', label='Mr. Hi')
        ax1.scatter(embeddings[:,0].detach().numpy()[idx_of], embeddings[:,1].detach().numpy()[idx_of], c = 'blue', label='Officer')
        ax1.scatter(archetypes[0,:].detach().numpy(), archetypes[1,:].detach().numpy(), marker = '^', c = 'black')
        ax1.annotate('Mr. Hi', embeddings[Mr_Hi,:])
        ax1.annotate('Officer', embeddings[John_A, :])
        ax1.legend()
        ax1.set_title(f"Latent space after {iterations} iterations")
        #Plotting learning curve
        ax2.plot(losses)
        ax2.set_title("Loss")
    plt.show()



