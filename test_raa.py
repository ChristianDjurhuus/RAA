
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM, LSMAA
from src.models.train_KAA_module import KAA
from src.models.calcNMI import calcNMI

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx
import archetypes as arch

k=3
n = 100
d = 2
adj_m, z, A, Z_true, beta, partition = main(alpha=0.2, k=k, dim=d, nsamples=n, rand=True)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

raa = DRRAA(k=k,
            d=d,
            sample_size=.3,
            data=edge_list,
            data_type = "edge list",
            link_pred = True,
            seed_split=1,
            seed_init=1
)
raa.train(iterations=10000, LR=0.1)

kaa = KAA(k=k,
          data=edge_list,
          type='jaccard',
          link_pred=True,
          seed_split =1,
          seed_init=1
)
kaa.train(iterations=10000, LR=0.1)


