import torch
import numpy as np
import networkx as nx


PATH = "...."

edge_list = np.genfromtxt("data/raw/....", delimiter=",")

org_sparse_i = edge_list[0, :]
org_sparse_j = edge_list[1, :]


#remove 50% while maintaining residual network is connected
N = max(map(max, edge_list))
num_samples = edge_list.shape[0]//2

nx_edgelist = [(int(edge_list[0,i]),int(edge_list[1,i])) for i in range(edge_list.shape[1])]
status=False

adj_m = nx.from_edgelist(nx_edgelist)

while status == False:
    status = True
    rm_indices = torch.multinomial(input=torch.arange(0, float(edge_list.shape[0])), num_samples=num_samples,
                                replacement=False)
    G = nx.from_edgelist(nx_edgelist)

    #Check for components 
    if nx.number_connected_components(G) > 1:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0]).copy()

    #Check if residual network remains connected 
    for i in range(rm_indices.shape[0]):
        G.remove_edge(int(edge_list[:,rm_indices[i]][0]), int(edge_list[:,rm_indices[i]][1]))
        if nx.number_connected_components(G) > 1:
            print('Did not manage to keep residual network connected')
            status = False
            break
    
#Sample negative links

nonlinks = (adj_m==0).nonzero()

negative_indices = torch.multinomial(input=torch.arange(0, float(nonlinks.shape[0])), num_samples=num_samples,
                                replacement=False)

#creating training set with removed edges
altered_adj_m = nx.adjacency_matrix(G)


sparse_i_rm = edge_list[:,rm_indices][0,:]
sparse_j_rm = edge_list[:,rm_indices][1,:]
non_sparse_i = nonlinks[negative_indices,:][:,0]
non_sparse_j = nonlinks[negative_indices,:][:,1]
sparse_i = altered_adj_m.tocoo().row
sparse_j = altered_adj_m.tocoo().col 


np.savetxt(f'data/train_masks/{PATH}/non_sparse_i.txt', non_sparse_i.numpy())
np.savetxt(f'data/train_masks/{PATH}/non_sparse_j.txt', non_sparse_j.numpy())
np.savetxt(f'data/train_masks/{PATH}/sparse_i_rem.txt', sparse_i_rm.numpy())
np.savetxt(f'data/train_masks/{PATH}/sparse_j_rem.txt', sparse_j_rm.numpy())
np.savetxt(f'data/train_masks/{PATH}/sparse_i.txt', sparse_i)
np.savetxt(f'data/train_masks/{PATH}/sparse_j.txt', sparse_j)
np.savetxt(f'data/train_masks/{PATH}/org_sparse_i.txt', org_sparse_i)
np.savetxt(f'data/train_masks/{PATH}/org_sparse_j.txt', org_sparse_j)
