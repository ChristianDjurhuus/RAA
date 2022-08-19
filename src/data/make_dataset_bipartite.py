import torch
import numpy as np
import networkx as nx
import scipy.sparse as ss
from networkx.algorithms import bipartite


PATH = "2020_congress"

adj_m = np.genfromtxt("data/raw/2020_congress/2020_congress_graph.csv", delimiter=",")[1:,1:]
missing_data = np.genfromtxt("data/raw/2020_congress/2020_congress_missing_data_idx.csv", delimiter=",", dtype=int)[1:,1:]
edge_list_i = ss.coo_matrix(adj_m).row
edge_list_j = ss.coo_matrix(adj_m).col
edge_list = np.vstack((edge_list_i,edge_list_j))


org_sparse_i = edge_list[0, :]
org_sparse_j = edge_list[1, :]


#remove 50% while maintaining residual network is connected

num_samples = edge_list.shape[1]//2

torch.manual_seed(1)
status=False

while status == False:
    status = True
    rm_indices = torch.multinomial(input=torch.arange(0, float(edge_list.shape[1])), num_samples=num_samples,
                                replacement=False)
    B = nx.bipartite.from_biadjacency_matrix(ss.coo_matrix(adj_m))

    #Check for components 
    if nx.number_connected_components(B) > 1:
        Bcc = sorted(nx.connected_components(B), key=len, reverse=True)
        B = B.subgraph(Bcc[0]).copy()

    #Check if residual network remains connected 
    for i in range(rm_indices.shape[0]):
        B.remove_edge(int(edge_list[:,rm_indices[i]][0]), 442+int(edge_list[:,rm_indices[i]][1]))
        if nx.number_connected_components(B) > 1:
            print('Did not manage to keep residual network connected')
            status = False
            break
    
#Sample negative links
all_zeros = (adj_m==0)
#Remove missing data!
all_zeros[missing_data[:,1],missing_data[:,0]] = False
nonlinks = all_zeros.nonzero()
nonlinks_i = nonlinks[0]
nonlinks_j = nonlinks[1]
negative_indices = torch.multinomial(input=torch.arange(0, float(nonlinks_i.shape[0])), num_samples=num_samples,replacement=False)

#creating training set with removed edges
altered_adj_m = nx.bipartite.biadjacency_matrix(B,row_order=np.arange(442),column_order=np.arange(442,442+252))


sparse_i_rm = edge_list[:,rm_indices][0,:]
sparse_j_rm = edge_list[:,rm_indices][1,:]
non_sparse_i = nonlinks_i[negative_indices]
non_sparse_j = nonlinks_j[negative_indices]
sparse_i = altered_adj_m.tocoo().row
sparse_j = altered_adj_m.tocoo().col 


np.savetxt(f'data/train_masks/{PATH}/non_sparse_i.txt', non_sparse_i)
np.savetxt(f'data/train_masks/{PATH}/non_sparse_j.txt', non_sparse_j)
np.savetxt(f'data/train_masks/{PATH}/sparse_i_rem.txt', sparse_i_rm)
np.savetxt(f'data/train_masks/{PATH}/sparse_j_rem.txt', sparse_j_rm)
np.savetxt(f'data/train_masks/{PATH}/sparse_i.txt', sparse_i)
np.savetxt(f'data/train_masks/{PATH}/sparse_j.txt', sparse_j)
np.savetxt(f'data/train_masks/{PATH}/org_sparse_i.txt', org_sparse_i)
np.savetxt(f'data/train_masks/{PATH}/org_sparse_j.txt', org_sparse_j)
