'''
This script runs node classification on synthetic data with Louvain communities.
'''
from src.models.train_DRRAA_module import DRRAA
from src.data.synthetic_data import main
from src.visualization.visualize import Visualization
from src.features.link_prediction import KNeighborsClassifier

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import networkx as nx
import community as community_louvain

#make data
k=3
d=2
nsamples=100
alphas = [0.2, 1, 5]
iter = 10000
n_neighbours = 3

means = {}
confs = {}
stds = {}

for alpha in alphas:
    #set seed
    seed = 1999
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    adj_m, z, A, Z, beta, partition = main(alpha=alpha, k=k, dim=d, nsamples=nsamples, rand=False)
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    nx.set_node_attributes(Graph, partition, 'group')
    RAA = DRRAA(k=k,
                d=d,
                sample_size=1,  # Without random sampling
                data=Graph,
                data_type='networkx',
                link_pred=False)
    RAA.train(iterations=iter, LR=0.01)
    RAA.decision_boundary_knn('group', n_neighbors=n_neighbours, filename=f'decision_boundary_knn_alpha_{alpha}.png')
    RAA.plot_latent_and_loss(iter,cmap=partition,file_name=f'recreate_synthetic_alpha_{alpha}.png')
    means[alpha], confs[alpha], stds[alpha] = RAA.KNeighborsClassifier('group', n_neighbours=n_neighbours)
    confs[alpha] = np.array(list(confs[alpha]))-means[alpha]
print('Means as values, alphas as keys:\n',means)
print('\n\nConfidence intervals as values, alphas as keys:\n',confs)
print('\n\nStds as values, alphas as keys:\n',stds)
