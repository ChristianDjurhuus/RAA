import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as f
from collections import defaultdict
from sklearn import metrics
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.colors as colors

####################
## Synthetic data ##
####################

def synthetic_data(k, dim, alpha, nsamples):
    '''
    Randomly uniformly sample of data within a predefined polytope ((k-1)-simplex) using the Dirichlet distribution
        k: number of archetypes
        dim: The dimensions of the final latent space 
        alpha: parameter in dirichlet distribution
        nsamples: number of samples
    '''

    alpha = [alpha for i in range(k)]
    A = np.zeros((dim, k))
    #A =  np.array([[12., 13.,  9.],
    #    [18.,  6., 12.],
    #    [14.,  7., 16.]])
    for i in range(k):
        A[:,i] = np.random.randint(20, size=dim).reshape(dim,)
    
    
    Z = np.zeros((k, nsamples))
    for i in range(nsamples):
        Z[:,i] = np.random.dirichlet(alpha)

    A = torch.from_numpy(A).float() #Should this sum to one?
    #A = f.softmax(A, dim=1)
    Z = torch.from_numpy(Z).float()
    return np.matmul(A, Z).T, A, Z

def logit2prob(logit):
    '''
    utils function //
    Convert logit to probability
    '''
    odds = torch.exp(logit)
    probs = (odds) / (1+odds)
    return probs

def convert(a):
    '''
    Utils function //
    Convert adjacency matrix to edgelist
    '''
    edge_list = np.zeros((2, int(sum(sum(a)))))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
                       if a[i][j]== 1:
                           edge_list[0, i] = i
                           edge_list[1, j] = j
    edge_list = edge_list[edge_list[0, :].argsort(), :]
    return edge_list


def generate_network_bias(A, Z, k, d, nsamples, rand = False):
    ''' Generate adj matrix, Undirected case & without dimensionality reduction
            Z: samples drawn from dirichlet distribution
            A: Archetypes
            rand: Random effects
    '''
    #A = torch.from_numpy(A).float() #Should this sum to one?
    #A = f.softmax(A, dim=1)
    #Z = torch.from_numpy(Z).float() #Already sums to one
    if rand:
        #Trying to replicate natural sparcity
        r1=-5
        r2=5
        a = 1
        b = nsamples
        beta = torch.FloatTensor(a, b).uniform_(r1, r2).reshape(-1)
    else:
        beta = torch.ones(nsamples)
        #dim_matrix = torch.rand(d, k)
        #dim_matrix = torch.diag(torch.tensor([40,40,40])).float()

    beta_matrix = beta.unsqueeze(1) + beta
    beta_matrix = beta_matrix/2
    #M = torch.matmul(dim_matrix, torch.matmul(A, Z)).T # (N x K)
    M = torch.matmul(A, Z).T
    z_dist = ((M.unsqueeze(1) - M + 1e-06)**2).sum(-1)**0.5 # (N x N)
    theta = beta_matrix - z_dist #beta_matrix - z_dist # (N x N) - log_odds
    #theta = 1 - z_dist
    probs = logit2prob(theta)
    adj_m = torch.bernoulli(probs) # bernoullig distribution to get links
    #Making adjacency matrix symmetric using upper triangle
    triu = torch.triu(adj_m)
    adj_m = triu + triu.T - torch.diag(torch.diagonal(adj_m))
    adj_m = adj_m - torch.diag(torch.diagonal(adj_m))
    return adj_m, beta

def ideal_prediction(adj_m, A, Z, beta, test_size = 0.5):
        '''
        A: Arcetypes
        Z: sampled datapoints
    '''
        num_samples = round(test_size * 0.5* (adj_m.shape[0] * (adj_m.shape[0] - 1)))
        idx_i_test = torch.multinomial(input=torch.arange(0, float(adj_m.shape[0])), num_samples=num_samples,
                                replacement=True)
        #Only sample upper corner
        idx_j_test = torch.multinomial(input=torch.arange(0, float(adj_m.shape[1])), num_samples=num_samples,
                                        replacement=True)

        adj_m = adj_m.clone().detach()

        value_test = adj_m[idx_i_test, idx_j_test].numpy()

        M_i = torch.matmul(A, Z[:, idx_i_test]).T
        M_j = torch.matmul(A, Z[:, idx_j_test]).T
        z_pdist_test = ((M_i - M_j + 1e-06) ** 2).sum(-1) ** 0.5
        theta = (beta[idx_i_test] + beta[idx_j_test] - z_pdist_test)
        rate = torch.exp(theta)
        fpr, tpr, threshold = metrics.roc_curve(value_test, rate.cpu().data.numpy())
        auc_score = metrics.roc_auc_score(value_test, rate.cpu().data.numpy())
        return auc_score, fpr, tpr


def get_clusters(adj_m):
    G = nx.from_numpy_matrix(adj_m.numpy())
    partition = community_louvain.best_partition(G)
    return partition

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    This function takes part of a matplotlib colormap and uses it. (we didnt like too white values)
    thanks to: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)

def main(alpha, k, dim, nsamples, rand):
    synth_data, A, Z = synthetic_data(k, dim, alpha, nsamples)
    adj_m, beta = generate_network_bias(A, Z, k, dim, nsamples, rand)

    #Removing disconnected components
    G = nx.from_numpy_matrix(adj_m.numpy())

    if nx.number_connected_components(G) > 1:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
        delete_Z = []
        for i in range(len(Gcc)):
            if Gcc[i] == Gcc[0]:
                continue
            else:
                for j in range(len(Gcc[i])):
                   delete_Z.append(list(Gcc[i])[j])
        mask = torch.ones((Z.shape[0],Z.shape[1]), dtype=torch.bool)
        mask[:, delete_Z] = False
        Z = Z[mask].reshape(k, mask.shape[1]-len(delete_Z))

        mask_adj = torch.ones((adj_m.shape[0],adj_m.shape[1]), dtype=torch.bool)
        mask_adj[:, delete_Z] = False
        mask_adj[delete_Z,:] = False
        adj_m = adj_m[mask_adj].reshape(adj_m.shape[0] - len(delete_Z),adj_m.shape[1] - len(delete_Z))

        synth_data = torch.matmul(A, Z).T
    
    #label_map = {x: i for i, x in enumerate(G.nodes)}
    #G = nx.relabel_nodes(G, label_map)


    #Louvain partition
    partition = get_clusters(adj_m)

    #Calculating density
    xy = np.vstack((synth_data[:,0].numpy(), synth_data[:,1].numpy()))
    z = gaussian_kde(xy)(xy)
    mpl.rcParams['font.family'] = 'Times New Roman'
    cmap = plt.get_cmap('RdPu')
    cmap = truncate_colormap(cmap, 0.2, 1)
    if dim == 3:
        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(synth_data[:, 0], synth_data[:, 1], synth_data[:, 2], c=z, cmap=cmap)
        ax.scatter(A[0, :], A[1, :], A[2, :], marker='^', c='black', label="Archetypes")
        ax.set_title(f"True Latent Space (alpha={alpha})")
        fig.colorbar(sc, label="Density")
    else:
        fig, ax = plt.subplots(dpi=500)
        sc = ax.scatter(synth_data[:, 0], synth_data[:, 1], c=z, cmap=cmap)
        #ax.scatter(synth_data[:, 0], synth_data[:, 1], c=list(partition.values()), cmap='Set2')
        ax.scatter(A[0, :], A[1, :], marker='^', c='black', label="Archetypes")
        #ax.set_title(f"True Latent Space (alpha={alpha})")
        fig.colorbar(sc, label="Density")
    ax.legend()
    plt.savefig(f'true_latent_space_{alpha}.png',dpi=500)
    #plt.show()

    plt.figure(dpi=500)
    plt.imshow(adj_m, cmap = 'hot', interpolation='nearest')
    #plt.title(f"Adjacency matrix ({alpha})")
    plt.savefig(f'synt_adjacency_{alpha}.png', dpi=500)
    #plt.show()

    fig, ax = plt.subplots(dpi=500)
    ax.scatter(synth_data[:, 0], synth_data[:, 1], c=list(partition.values()), cmap='tab10')
    ax.scatter(A[0, :], A[1, :], marker='^', c='black', label="Archetypes")
    #ax.set_title(f"True_latent_space_louvain.png", dpi=500)
    plt.savefig(f"True_latent_space_louvain_{alpha}.png", dpi=500)
    #plt.show()

    return adj_m, z, A, Z, beta, partition

    






