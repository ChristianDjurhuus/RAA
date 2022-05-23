'''
This script runs node classification on synthetic data with Louvain communities.
'''
from src.models.train_DRRAA_module import DRRAA
from src.data.synthetic_data import main
from src.visualization.visualize import Visualization

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import networkx as nx
import community as community_louvain

#set seed
seed = 1999
torch.random.manual_seed(seed)
np.random.seed(seed)

#make data
k=3
d=2
nsamples=100
alphas = [0.2, 1, 5]
iter = 10000
for alpha in alphas:
    adj_m, z, A, Z, beta, partition = main(alpha=alpha, k=k, dim=d, nsamples=nsamples, rand=False)
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    RAA = DRRAA(k=k,
                d=d,
                sample_size=1,  # Without random sampling
                data=Graph,
                data_type='networkx',
                link_pred=False)
