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
d = 2
N = 100

AUC_raa = []
AUC_raa_ngating = []
NMIs_ngating = []
NMIs_raa = []
ks = 3 #should be 2 or more
true_k = 3

np.random.seed(seed)
adj_m, z, A, Z_true = main(alpha=0.2,k=true_k,N=N)  # z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

edge_list = torch.from_numpy(edge_list).long()
for k in range(2,ks+1):
    raa = DRRAA(k=k,
                d=d,
                sample_size=1,
                data=edge_list)
                #data_type='Adjacency matrix')

    # Training models
    iter = 5000
    raa.train(iterations=iter)

    raa_auc_score, _, _ = raa.link_prediction()



    AUC_raa.append(raa_auc_score)


    # Determining NMI
    Z_raa = f.softmax(raa.Z, dim=0)
    NMIs_raa.append(calcNMI(Z_raa, Z_true).item())

mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(np.arange(2,ks+1), AUC_raa, label='RAA')
ax.vlines(true_k, 0,1,colors='red',label='True k')
ax.set_xlabel("k")
ax.set_ylabel("AUC score")
ax.set_title("AUC score with hidden archetype")
ax.legend()
plt.savefig("AUCs_set_k.png")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(np.arange(2,ks+1), NMIs_raa, label='NMI')
ax.vlines(true_k, 0,1,colors='red',label='True k')
ax.set_xlabel("k")
ax.set_title("The NMI with hidden archetype")
ax.set_ylabel("NMI score")
ax.legend()
plt.savefig("NMIs_set_k.png")
plt.show()