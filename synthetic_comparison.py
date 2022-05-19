from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
from src.data.synthetic_data import main
from src.models.calcNMI import calcNMI
from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_KAA_module import KAA
from src.models.train_DRRAA_ngating import DRRAA_ngating

import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as f

seed = 4
torch.random.manual_seed(seed)
np.random.seed(seed)
k = 3
d = 2
N = 100

AUC_raa = []
AUC_lsm = []
AUC_kaa = []
AUC_raa_nre = []
AUC_kaa_j = []
AUC_kaa_n = []
AUC_raa_ng = []

NMIs = []
NMIs_nre = []
NMIs_ng = []

alpha_values =  [0.2, 1, 5]
for alpha in alpha_values:
    adj_m, z, A, Z_true = main(alpha) #z is cmap
    G = nx.from_numpy_matrix(adj_m.numpy())

    temp = [x for x in nx.generate_edgelist(G, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)): 
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    edge_list = torch.from_numpy(edge_list).long()
    raa = DRRAA(k=k,
                d=d, 
                sample_size=1, #Without random sampling
                data=edge_list)

    lsm = LSM(latent_dim=d, 
              sample_size=1, #Without random sampling
              data=edge_list)

    kaa_j = KAA(k=k, data=adj_m, type='jaccard')

    kaa_n = KAA(k=k, data=adj_m, type='normalised_x')

    #w/o random effects
    raa_nre = DRRAA_nre(k=k,
                d=d, 
                sample_size=1, #Without random sampling
                data=edge_list)
    
    raa_ng = DRRAA_ngating(k=k,
                d=d, 
                sample_size=1, #Without random sampling
                data=edge_list)

    #Training models
    iter = 10000
    raa.train(iterations=iter)
    lsm.train(iterations=iter)
    raa_nre.train(iterations=iter)
    kaa_j.train(iterations=iter)
    kaa_n.train(iterations=iter)
    raa_ng.train(iterations=iter)

    raa_auc_score, _, _ = raa.link_prediction()
    lsm_auc_score, _, _ = lsm.link_prediction()
    raa_nre_score, _, _ = raa_nre.link_prediction()
    kaa_j_auc_score, _, _ = kaa_j.link_prediction()
    kaa_n_auc_score, _, _ = kaa_n.link_prediction()
    raa_ng_auc_score, _, _ = raa_ng.link_prediction()


    AUC_raa.append(raa_auc_score)
    AUC_lsm.append(lsm_auc_score)
    AUC_raa_nre.append(raa_nre_score)
    AUC_kaa_j.append(kaa_j_auc_score)
    AUC_kaa_n.append(kaa_n_auc_score)
    AUC_raa_ng.append(raa_ng_auc_score)

    #Determining NMI
    Z = f.softmax(raa.Z, dim=0)
    Z_nre = f.softmax(raa_nre.Z, dim=0)
    Z_ng = f.softmax(raa_ng.Z, dim=0)
    NMIs.append(calcNMI(Z, Z_true).item())
    NMIs_nre.append(calcNMI(Z_nre, Z_true).item())
    NMIs_ng.append(calcNMI(Z_ng, Z_true).item())


    

mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(alpha_values, AUC_raa, '-o', label='RAA')
ax.plot(alpha_values, AUC_raa_nre, '-o', label='RAA w/o random effects')
ax.plot(alpha_values, AUC_raa_ng, '-o', label='RAA w/o gatingn')
ax.plot(alpha_values, AUC_lsm, '-o', label='LSM')
ax.plot(alpha_values, AUC_kaa_j, '-o', label='KAA - Jaccard')
ax.plot(alpha_values, AUC_kaa_n, '-o', label='KAA - Normalized')
ax.set_xlabel("Alpha values")
ax.set_ylabel("AUC score")
ax.set_title("AUC score with varying alpha values")
ax.legend()
plt.savefig("AUCs_comp_seed.png")
#plt.show()

fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(alpha_values, NMIs, '-o', label='NMIs - RAA')
ax.plot(alpha_values, NMIs_nre, '-o', label='NMIs - RAA without random effects')
ax.plot(alpha_values, NMIs_ng, '-o', label='NMIs - RAA without gating')
ax.set_xlabel("alpha value")
ax.set_title("The NMI with different alpha values")
ax.set_ylabel("score")
ax.legend()
plt.savefig("NMIs_comp_seed.png")
#plt.show()


