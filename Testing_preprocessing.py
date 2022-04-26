from src.models.train_DRRAA_module import DRRAA
from src.models.train_DRRAA_ngating import DRRAA_ngating
from src.models.train_KAA import KAA
from src.models.synthetic_data import main
from src.models.calcNMI import calcNMI
#from src.models.train_DRRAA_nre import DRRAA_nre

import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as f

seed = 38000
torch.random.manual_seed(seed)
d = 3
N = 100

AUC_raa = []
AUC_raa_ngating = []
NMIs_ngating = []
NMIs_raa = []

np.random.seed(seed)
adj_m, z, A, Z_true = main(alpha=0.2,k=3,N=N)  # z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())

raa = DRRAA(k=3,
            d=d,
            sample_size=1,
            data=G,
            data_type='Networkx')

# Training models
iter = 5000
raa.train(iterations=iter)

raa_auc_score, _, _ = raa.link_prediction()



print('AUC:',raa_auc_score)


# Determining NMI
Z_raa = f.softmax(raa.Z, dim=0)
print('NMI:',(calcNMI(Z_raa, Z_true).item()))
