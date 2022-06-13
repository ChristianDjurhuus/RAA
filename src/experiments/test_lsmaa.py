from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSMAA, LSM
from src.models.train_KAA_module import KAA
from src.models.calcNMI import calcNMI
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx
from py_pcha import PCHA
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1)
torch.manual_seed(1)

top10 = np.arange(10)

#create data before the runs to make sure we test initialisations of models:
real_alpha = 0.2
K = 3
n = 100
d = 2
adj_m, z, A, Z_true, beta, partition = main(alpha=real_alpha, k=K, dim=d, nsamples=n, rand=False)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#set test and train split seed. We want the same train and test split in order to know that differences
#are because of inits.
seed_split = 42

lsmaa_nmi = LSMAA(d=2,
              k=4,
              sample_size=1,
              data=edge_list,
              data_type="Edge list",
              link_pred=True,
              seed_split=seed_split,
              seed_init=1
              )
lsmaa_nmi.train(iterations=10)
lsmaa_auc, _, _ = lsmaa_nmi.link_prediction()
print(lsmaa_auc)