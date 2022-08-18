import torch
import numpy as np
import networkx as nx
import scipy.sparse as ss
from networkx.algorithms import bipartite


PATH = "2020_congress"

adj_m = np.genfromtxt("data/raw/2020_congress/2020_congress_graph.csv", delimiter=",")[1:,1:]
edge_list_i = ss.coo_matrix(adj_m).row
#edge_list_i = [str(edge_list_i[i]) for i in range(len(edge_list_i))]
edge_list_j = ss.coo_matrix(adj_m).col
edge_list_j = [adj_m.shape[0]+edge_list_j[j] for j in range(len(edge_list_j))]
edge_list = np.vstack((edge_list_i,edge_list_j))
org_sparse_i = edge_list[0, :]
org_sparse_j = edge_list[1, :]


#remove 50% while maintaining residual network is connected
N = max(map(max, edge_list))
num_samples = edge_list.shape[1]//2

nx_edgelist = [(int(edge_list[0,i]),int(edge_list[1,i])) for i in range(edge_list.shape[1])]
status=False

while status == False:
    status = True
    rm_indices = torch.multinomial(input=torch.arange(0, float(edge_list.shape[1])), num_samples=num_samples,
                                replacement=False)

    G = nx.Graph()
     # Add nodes with the node attribute "bipartite"
    G.add_nodes_from(np.arange(adj_m.shape[0]), bipartite=0)
    G.add_nodes_from(np.arange(start=adj_m.shape[0], stop=adj_m.shape[0]+adj_m.shape[1]), bipartite=1)
    # Add edges only between nodes of opposite node sets
    G.add_edges_from(list(zip(edge_list[0,:],edge_list[1,:])))
    #G = nx.from_edgelist(nx_edgelist)

    #Check for components 
    if nx.number_connected_components(G) > 1:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0]).copy()

    #Check if residual network remains connected 
    for i in range(rm_indices.shape[0]):
        G.remove_edge(int(edge_list[:,rm_indices[i]][0]), int(edge_list[:,rm_indices[i]][1]))
        print(i)
        if nx.number_connected_components(G) > 1:
            print('Did not manage to keep residual network connected')
            status = False
            break
    
#Sample negative links

nonlinks = np.array((adj_m==0).nonzero())

negative_indices = torch.multinomial(input=torch.arange(0, float(nonlinks.shape[1])), num_samples=num_samples,replacement=False)

#creating training set with removed edges
altered_adj_m = nx.adjacency_matrix(G)


sparse_i_rm = edge_list[:,rm_indices][0,:]
sparse_j_rm = edge_list[:,rm_indices][1,:]
non_sparse_i = nonlinks[:,negative_indices][0,:]
non_sparse_j = nonlinks[:,negative_indices][1,:]
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
