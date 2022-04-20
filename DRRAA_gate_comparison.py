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

seed = 42
torch.random.manual_seed(seed)
k = 3
d = 3
N = 100

AUC_raa = []
AUC_raa_ngating = []
NMIs_ngating = []
NMIs_raa = []
ks = 12
for k in range(2,ks):
    adj_m, z, A, Z_true = main(alpha=0.2)  # z is cmap
    G = nx.from_numpy_matrix(adj_m.numpy())

    temp = [x for x in nx.generate_edgelist(G, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)):
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    edge_list = torch.from_numpy(edge_list).long()
    raa = DRRAA(k=k,
                d=d,
                sample_size=1,
                data=edge_list)

    # w/o random effects
    raa_ngating = DRRAA_ngating(k=k,
                        d=d,
                        sample_size=1,  # Without gating function
                        data=edge_list)

    # Training models
    iter = 5000
    raa.train(iterations=iter)
    raa_ngating.train(iterations=iter)

    raa_auc_score, _, _ = raa.link_prediction()

    raa_ngating_score, _, _ = raa_ngating.link_prediction()


    AUC_raa.append(raa_auc_score)
    AUC_raa_ngating.append(raa_ngating_score)


    # Determining NMI
    Z_raa = f.softmax(raa.Z, dim=0)
    NMIs_raa.append(calcNMI(Z_raa, Z_true).item())

    Z_ngating = f.softmax(raa_ngating.Z, dim=0)
    NMIs_ngating.append(calcNMI(Z_ngating, Z_true).item())

mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(np.arange((2,ks)), AUC_raa, label='RAA')
ax.plot(np.arange((2,ks)), AUC_raa_ngating, label='RAA w/o gating function')
ax.set_xlabel("k")
ax.set_ylabel("AUC score")
ax.set_title("AUC score with varying number of archtypes")
ax.legend()
plt.savefig("AUCs.png")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(np.arange((2,ks)), NMIs_raa, label='NMIs')
ax.plot(np.arange((2,ks)), NMIs_ngating, label='NMIs no gating')
ax.set_xlabel("k")
ax.set_title("The NMI with varying number of archtypes")
ax.set_ylabel("NMI score")
ax.legend()
plt.savefig("NMIs.png")
plt.show()

