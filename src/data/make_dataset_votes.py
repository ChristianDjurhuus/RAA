import numpy as np
import torch
import networkx as nx
import scipy.sparse as ss
from tqdm import tqdm

PATH = '2020_congress_uni'

adj_m = torch.from_numpy(np.genfromtxt("data/raw/2020_congress/2020_congress_graph_unipartite.csv", delimiter=",")[1:,1:])
edge_list_i = ss.coo_matrix(adj_m).row
edge_list_j = ss.coo_matrix(adj_m).col
edge_list = np.vstack((edge_list_i,edge_list_j))


org_sparse_i = np.array(edge_list[0, :])
org_sparse_j = np.array(edge_list[1, :])
org_values = ss.coo_matrix(adj_m).data

np.savetxt(f'data/train_masks/{PATH}/org_sparse_i.txt', org_sparse_i)
np.savetxt(f'data/train_masks/{PATH}/org_sparse_j.txt', org_sparse_j)
np.savetxt(f'data/train_masks/{PATH}/org_values.txt', org_values)