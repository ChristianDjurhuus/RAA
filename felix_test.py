from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
from src.models.train_KAA_module import KAA
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
from src.data.synthetic_data import main
import networkx as nx

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

##############################
# Synthetic model comparison #
##############################


#Data creation
alpha = 0.2
K = 3
n = 100
d = 2
adj_m, z, A, Z_true = main(alpha=alpha, k=K, dim=d, nsamples=n)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#edge_list = torch.from_numpy(edge_list).long()

#Defining models
iter = 10000

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

raa = DRRAA(d=d,k=3,
            sample_size=1, #Without random sampling
            data=edge_list,
            data_type="edge list",
            link_pred=True)

kaa = KAA(k=3,
          data=adj_m.numpy(),
          type = "jaccard",
          link_pred = True,
          )

lsm = LSM(d=2,
          sample_size=1,
          data=edge_list,
          data_type = "Edge list",
          link_pred=True)

raa.train(iterations=iter)
raa_auc, _, _ = raa.link_prediction()
kaa.train(iterations=iter)
kaa_auc, _, _ = kaa.link_prediction()
lsm.train(iterations=iter)
lsm_auc, _, _ = lsm.link_prediction()

print(raa_auc, lsm_auc, kaa_auc)