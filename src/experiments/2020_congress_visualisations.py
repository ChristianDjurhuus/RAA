import pandas as pd
import torch
from src.models.train_BDRRAA_module import BDRRAA
import networkx as nx
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.calcNMI import calcNMI
import matplotlib as mpl
from src.data.synthetic_data import truncate_colormap
import scipy

adj_m = np.array(pd.read_csv('data/raw/2020_congress/2020_congress_graph.csv', index_col=False))[:,1:]
#Gi = scipy.sparse.coo_matrix(G).row
#Gj = scipy.sparse.coo_matrix(G).col
#G = np.concatenate(Gi,Gj)
metadata = np.array(pd.read_csv('data/raw/2020_congress/2020_congress_metadata.csv', index_col=False))[:,1:]
missing_data = np.array(pd.read_csv('data/raw/2020_congress/2020_congress_missing_data_idx.csv', index_col=False))[:,1:]

temp = torch.from_numpy(adj_m - np.tril(adj_m))
adj_m = torch.from_numpy(adj_m)
links = torch.nonzero(temp)
N = adj_m.shape[0]
num_samples = links.shape[0]//2

status = False
while status == False:
    status = True
    rm_indices = torch.multinomial(input=torch.arange(0, float(links.shape[0])), num_samples=num_samples,
                                    replacement=False)
    #Check if residual network remains connected
    G = nx.from_numpy_array(adj_m.numpy())
    for i in range(rm_indices.shape[0]):
        G.remove_edge(int(links[rm_indices[i],:][0]), int(links[rm_indices[i],:][1]))
        if nx.number_connected_components(G) > 1:
            print('Did not manage to keep residual network connected')
            status = False
            break
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




model = BDRRAA(k=3,
                    d=2,
                    sample_size=0.5,
                    data=sparse_i,
                    data2=sparse_j, non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rm=sparse_i_rm,sparse_j_rem=sparse_j_rm,
                    data_type = "sparse",
        )
model.train(iterations=10, print_loss=True)



