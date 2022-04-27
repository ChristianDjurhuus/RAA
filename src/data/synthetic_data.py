import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as f
from collections import defaultdict
import pickle
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
        dim_matrix = torch.rand(d, k)
    else:
        beta = torch.ones(nsamples)
        #dim_matrix = torch.rand(d, k)
        #dim_matrix = torch.diag(torch.tensor([40,40,40])).float()

    beta_matrix = beta.unsqueeze(1) + beta
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
    return adj_m

def main(alpha, k, dim, nsamples):
    seed = 1984
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    #d = 2
    synth_data, A, Z = synthetic_data(k, dim, alpha, nsamples)
    adj_m = generate_network_bias(A, Z, k, dim, nsamples, rand=False)

    #Calculating density
    xy = np.vstack((synth_data[:,0].numpy(), synth_data[:,1].numpy()))
    z = gaussian_kde(xy)(xy)
    mpl.rcParams['font.family'] = 'Times New Roman'
    if dim == 3:
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(synth_data[:, 0], synth_data[:, 1], synth_data[:, 2], c=z, cmap='viridis')
        ax.scatter(A[0, :], A[1, :], A[2, :], marker='^', c='black', label="Archetypes")
        ax.set_title(f"True Latent Space (alpha={alpha})")
        fig.colorbar(sc, label="Density")
    else:
        fig, ax = plt.subplots(dpi=100)
        sc = ax.scatter(synth_data[:, 0], synth_data[:, 1], c=z, cmap='viridis')
        ax.scatter(A[0, :], A[1, :], marker='^', c='black', label="Archetypes")
        ax.set_title(f"True Latent Space (alpha={alpha})")
        fig.colorbar(sc, label="Density")
    ax.legend()
    plt.show()
    #plt.savefig(f'true_latent_space_{alpha}.png')

    plt.figure(dpi=100)
    plt.imshow(adj_m, cmap = 'hot', interpolation='nearest')
    #plt.title(f"Adjacency matrix ({alpha})")
    #plt.savefig(f'synt_adjacency_{alpha}.png')
    plt.show()

    return adj_m, z, A, Z

    


if __name__ == "__main__":
    main(alpha=0.05, k=10, dim=2, nsamples=1000)


