import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as f
from collections import defaultdict
import pickle
####################
## Synthetic data ##
####################

def synthetic_data(k, alpha, nsamples):
    '''
    Randomly uniformly sample of data within a predefined polytope ((k-1)-simplex) using the Dirichlet distribution
        k: number of archetypes
        alpha: parameter in dirichlet distribution
        nsamples: number of samples
    '''

    alpha = [alpha for i in range(k)]
    A = np.zeros((k, k))
    for i in range(k):
        A[:,i] = np.random.rand(k,1).reshape(k,)

    Z = np.zeros((k, nsamples))
    for i in range(nsamples):
        Z[:,i] = np.random.dirichlet(alpha)

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

    A = torch.from_numpy(A).float() #Should this sum to one?
    A = f.softmax(A, dim=1)
    Z = torch.from_numpy(Z).float() #Already sums to one
    if rand:
        #Trying to replicate natural sparcity
        r1=-10
        r2= 2
        a = 1
        b = nsamples
        beta = torch.FloatTensor(a, b).uniform_(r1, r2).reshape(-1)
        #beta = torch.rand(nsamples)
        dim_matrix = torch.rand(d, k)
    else:
        beta = torch.zeros(nsamples)
        #dim_matrix = torch.eye((d, k))
        dim_matrix = torch.rand(d, k)
    
    beta_matrix = beta.unsqueeze(1) + beta
    M = torch.matmul(dim_matrix, torch.matmul(A, Z)).T # (N x K)
    z_dist = ((M.unsqueeze(1) - M + 1e-06)**2).sum(-1)**0.5 # (N x N)
    theta = beta_matrix - z_dist # (N x N) - log_odds
    probs = logit2prob(theta)
    adj_m = torch.bernoulli(probs) # bernoullig distribution to get links
    #Making adjacency matrix symmetric using upper triangle
    tril = torch.triu(adj_m)
    adj_m = tril + tril.T - torch.diag(torch.diagonal(adj_m))
    return adj_m

def main():
    d = 3
    k = 3
    alpha = 0.2
    nsamples=100
    synth_data, A, Z = synthetic_data(k, alpha, nsamples)
    adj_m = generate_network_bias(A, Z, k, d, nsamples, rand=True)

    #Calculating density
    xy = np.vstack((synth_data[:,1], synth_data[:,2]))
    z = gaussian_kde(xy)(xy)
    if synth_data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(synth_data[:, 0], synth_data[:, 1],
                    synth_data[:, 2], c=z, cmap='viridis')
        ax.scatter(A[0, :], A[1, :],
                    A[2, :], marker='^', c='black', label="Archetypes")
        ax.set_title(f"True Latent Space")
        fig.colorbar(sc, label="Density")
        ax.legend()
    plt.show()
    plt.close()

    plt.imshow(adj_m, cmap = 'hot', interpolation='nearest')
    plt.show()

    #edge_list = convert(adj_m)
    #with open("synth_data", "wb") as fp: #https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    #    pickle.dump(edge_list, fp)
    return adj_m, z 

    


if __name__ == "__main__":
    main()


