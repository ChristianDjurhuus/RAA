from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSMAA, LSM
from src.models.synthetic_data import main
from src.models.calcNMI import calcNMI


import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as f
import archetypes

seed = 42
torch.random.manual_seed(seed)
d = 3
N = 100

AUC_raa = []
AUC_lsm = []
AUC_lsmaa = []
NMIs_lsm = []
NMIs_raa = []
NMIs_lsmaa = []
k = 3
alphas = np.array([0.2,1,5])
A = np.array([[12., 13., 9.],
                  [18., 6., 12.],
                  [14., 7., 16.]])
for alpha in alphas:
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    adj_m, z, A, Z_true = main(alpha=alpha,k=k,N=N,A=A,d=d)  # z is cmap
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
                data=adj_m.numpy(),
                data_type="Adjacency matrix")

    # w/o random effects
    lsm = LSM(latent_dim=d, sample_size=1, data=adj_m.numpy(), data_type="Adjacency matrix")
    lsmaa = LSMAA(latent_dim=d,k=k, sample_size=1, data=adj_m.numpy(), data_type="Adjacency matrix")

    # Training models
    iter = 5000
    raa.train(iterations=iter)
    lsm.train(iterations=iter)
    lsmaa.train(iterations=iter)

    raa_auc_score, _, _ = raa.link_prediction()

    lsm_auc_score, _, _ = lsm.link_prediction()

    lsmaa_auc_score, _, _ = lsmaa.link_prediction()


    AUC_raa.append(raa_auc_score)
    AUC_lsm.append(lsm_auc_score)
    AUC_lsmaa.append(lsmaa_auc_score)


    # Determining NMI
    #raa
    Z_raa = f.softmax(raa.Z, dim=0)
    NMIs_raa.append(calcNMI(Z_raa, Z_true).item())

    #LSM
    lsm_z = lsm.latent_Z
    NMIs_lsm.append(calcNMI(lsm_z.T, Z_true).item())

    #LSM with AA (two step)
    aa = archetypes.AA(n_archetypes=k)
    lsmaa_z = aa.fit_transform(lsmaa.latent_Z.detach().numpy())
    latent_Z = torch.from_numpy(lsmaa_z).float()
    Z_lsmaa = f.softmax(latent_Z, dim=0)
    NMIs_lsmaa.append(calcNMI(Z_lsmaa.T, Z_true).item())

mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(alphas, AUC_raa, label='RAA')
ax.plot(alphas, AUC_lsm, label='LSM')
ax.plot(alphas, AUC_lsmaa, label='LSM w/ AA (2-step)')
ax.set_xlabel("alpha value")
ax.set_ylabel("AUC score")
ax.set_title("AUC score with varying alpha values")
ax.legend()
plt.savefig("AUCs_twostep_raa.png")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(alphas, NMIs_raa, label='NMIs RAA')
ax.plot(alphas, AUC_lsm, label='LSM')
ax.plot(alphas, NMIs_lsmaa, label='NMIs LSM w/ AA (2-step)')
ax.set_xlabel("alpha value")
ax.set_title("The NMI with varying alpha values")
ax.set_ylabel("NMI score")
ax.legend()
plt.savefig("NMIs_twostep_raa.png")
plt.show()