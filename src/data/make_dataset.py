from synthetic_data import main
import torch
import numpy as np
import networkx as nx

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

alpha=5
k=3
dim=2
nsamples=1000
rand=False

adj_m, z, A, Z, beta = main(alpha=alpha, k=k, dim=2, nsamples=1000, rand=rand)


#remove 50% while maintaining residual network is connected
#only sample from upper corner
temp = adj_m - torch.tril(adj_m)
links = torch.nonzero(temp)
N = adj_m.shape[0]
num_samples = links.shape[0]//2

rm_indices = torch.multinomial(input=torch.arange(0, float(links.shape[0])), num_samples=num_samples,
                                  replacement=False)

#Check if residual network remains connected
G = nx.from_numpy_array(adj_m.numpy())
for i in range(rm_indices.shape[0]):
    G.remove_edge(int(links[rm_indices[i],:][0]), int(links[rm_indices[i],:][1]))
    if nx.number_connected_components(G) > 1:
        print('Did not manage to keep residual network connected')

#sample negative links
temp += torch.tril(torch.ones(*temp.shape)) #only sample from upper corner
nonlinks = (temp==0).nonzero()

negative_indices = torch.multinomial(input=torch.arange(0, float(nonlinks.shape[0])), num_samples=num_samples,
                                  replacement=False)

#creating training set with removed edges
altered_adj_m = nx.adjacency_matrix(G)


sparse_i_rm = links[rm_indices,:][:,0]
sparse_j_rm = links[rm_indices,:][:,1]
non_sparse_i = nonlinks[negative_indices,:][:,0]
non_sparse_j = nonlinks[negative_indices,:][:,1]
sparse_i = altered_adj_m.tocoo().row
sparse_j = altered_adj_m.tocoo().col


np.savetxt('non_sparse_i.txt', non_sparse_i.numpy())
np.savetxt('non_sparse_j.txt', non_sparse_j.numpy())
np.savetxt('sparse_i_rem.txt', sparse_i_rm.numpy())
np.savetxt('sparse_j_rem.txt', sparse_j_rm.numpy())
np.savetxt('sparse_i.txt', sparse_i)
np.savetxt('sparse_j.txt', sparse_j)