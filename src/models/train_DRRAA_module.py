import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from torch_sparse import spspmm

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing

class DRRAA(nn.Module, Preprocessing, Link_prediction, Visualization):
    def __init__(self, k, d, sample_size, data, data_type = "Edge list", data_2 = None):
        # TODO Skal finde en måde at loade data ind på. CHECK
        # TODO Skal sørge for at alle classes for de parametre de skal bruge.
        # TODO Skal ha indført en train funktion/class
        # TODO SKal ha udvides visualiseringskoden
        # TODO Skal sikre at alt kommer på det rigtige device
        # TODO Skal ha lavet performancec check ()

        super(DRRAA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        Preprocessing.__init__(self, data = data, data_type = data_type, device = self.device, data_2 = data_2)
        self.edge_list, self.N = Preprocessing.convert_to_egde_list(self)
        Link_prediction.__init__(self, edge_list = self.edge_list)
        Visualization.__init__(self)

        self.input_size = (self.N, self.N)
        self.k = k
        self.d = d

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0], device = self.device))
        self.softplus = nn.Softplus()
        self.A = torch.nn.Parameter(torch.randn(self.d, self.k, device = self.device))
        #self.u, self.sigma, self.vt = torch.svd(torch.nn.Parameter(torch.randn(self.d, self.k)))
        #self.A = torch.nn.Parameter(self.sigma * self.vt)
        self.Z = torch.nn.Parameter(torch.randn(self.k, self.input_size[0], device = self.device))
        #self.Z = torch.nn.Parameter(torch.load("src/models/S_initial.pt"))
        self.G = torch.nn.Parameter(torch.randn(self.input_size[0], self.k, device = self.device))

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
        Z = F.softmax(self.Z, dim=0) #(K x N)
        G = torch.sigmoid(self.G) #Sigmoid activation function
        C = (Z.T * G) / (Z.T * G).sum(0) #Gating function
        #For the nodes without links
        beta = self.beta[sample_idx].unsqueeze(1) + self.beta[sample_idx] #(N x N)
        AZCz = torch.mm(self.A, torch.mm(torch.mm(Z[:,sample_idx], C[sample_idx,:]), Z[:,sample_idx])).T
        mat = torch.exp(beta-((AZCz.unsqueeze(1) - AZCz + 1e-06) ** 2).sum(-1) ** 0.5)
        z_pdist1 = (0.5 * torch.mm(torch.exp(torch.ones(sample_idx.shape[0], device = self.device).unsqueeze(0)),
                                                          (torch.mm((mat - torch.diag(torch.diagonal(mat))),
                                                                    torch.exp(torch.ones(sample_idx.shape[0], device = self.device)).unsqueeze(-1)))))
        #For the nodes with links
        AZC = torch.mm(self.A, torch.mm(Z[:, sample_idx],C[sample_idx, :])) #This could perhaps be a computational issue
        z_pdist2 = (self.beta[sparse_sample_i] + self.beta[sparse_sample_j] - (((( torch.matmul(AZC, Z[:, sparse_sample_i]).T - torch.mm(AZC, Z[:, sparse_sample_j]).T + 1e-06) ** 2).sum(-1))) ** 0.5).sum()

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
    seed = 4
    torch.random.manual_seed(seed)

    #A = mmread("data/raw/soc-karate.mtx")
    #A = A.todense()
    #G = nx.read_gml('data/raw/polblogs/polblogs.gml')
    #label_map = {x: i for i, x in enumerate(G.nodes)}
    #G = nx.relabel_nodes(G, label_map)
    #N = len(G.nodes())
    # Get the edge list
    #temp = [x for x in nx.generate_edgelist(G, data=False)]
    #edge_list = np.zeros((2, len(temp)))
    #for i in range(len(temp)):
    #    edge_list[0, i] = temp[i].split()[0]
    #    edge_list[1, i] = temp[i].split()[1]
    data1 = np.loadtxt("data/raw/github/sparse_i.txt", delimiter=",", unpack=False)
    data2 = np.loadtxt("data/raw/github/sparse_j.txt", delimiter=",", unpack=False)
    #data\raw\github\sparse_i.txt
    N = len(data1)
    #edge_list = []
    #edge_list.append(data1)
    #edge_list.append(data2)
    #print(f"edge_list shape: {np.shape(edge_list)}")
    #edge_list = torch.tensor(edge_list).long().cuda()
    edge_list = torch.zeros((len(data1), 2))
    for idx in range(len(data1)):
        edge_list[idx, 0] = data1[idx]
        edge_list[idx, 1] = data2[idx]
    edge_list = edge_list.long().cuda()
    print("data loaded")
    #Setting number of archetypes and dimensions of latent space
    k = 5
    d = 2

    link_pred = False

    if link_pred:
        num_samples = round(0.1*N)
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

    model = DRRAA(input_size = (N, N),
                    k=k,
                    d=d, 
                    sampling_weights=torch.ones(N), 
                    sample_size=round(0.025 * N),
                    edge_list=edge_list)

    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    
    losses = []
    iterations = 100
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
        plt.title("RAA model")
        plt.show()

    #Plotting latent space
    Z = F.softmax(model.Z, dim=0)
    G = torch.sigmoid(model.G)
    C = (Z.T * G) / (Z.T * G).sum(0)

    embeddings = torch.matmul(model.A, torch.matmul(torch.matmul(Z, C), Z)).T
    embeddings = embeddings.cpu()
    archetypes = torch.matmul(model.A, torch.matmul(Z, C))
    archetypes = archetypes.cpu()

    #labels = list(club_labels.values())
    #idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    #idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(Z.detach().cpu().numpy(), cmap="YlGnBu", cbar=False, ax=ax1)
    sns.heatmap(C.T.detach().cpu().numpy(), cmap="YlGnBu", cbar=False, ax=ax2)



    #import colorcet as cc
    #from matplotlib import cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from fast_histogram import histogram2d

    #cmap = cc.cm["YIOrBr"].copy()
    #cmap.set_bad(cmap.get_under())  # set the color for 0bounds = [[X[:, 0].min(), X[:, 0].max()], [X[:, 1].min(), X[:, 1].max()]]
    bounds = [[embeddings.detach().numpy()[:, 0].min(), embeddings.detach().numpy()[:, 0].max()], [embeddings.detach().numpy()[:, 1].min(), embeddings.detach().numpy()[:, 1].max()]]
    print(bounds)
    h = histogram2d(embeddings.detach().numpy()[:, 0], embeddings.detach().numpy()[:, 1], range = bounds, bins=round(np.sqrt(N)))
    #fig, ax = plt.subplots()
    #pcm = ax.pcolor(h,
    #    norm = colors.LogNorm(cmap='PuBu_r', shading='auto'))
    #fig.colorbar(pcm, ax = ax, extend = "max")
    plt.imshow(h, cmap = "OrRd", norm=colors.LogNorm())
    plt.axis('off')
    plt.colorbar()
    plt.show()
