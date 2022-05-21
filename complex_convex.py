'''
Estimate complex convex hull: 
    Ideal predictor: True 
    Run with synthetic data K = 8 
    Synthetic alpha = 0.05 (wip) 
    N = 100 (wip) 
    K = 2 .. 10 
    D = 2 
    CV = 5 
    seed = 1998 
    sample_size = 1 (wip) 
    Lr = 0.01 (default) 
    Iterations = 10,000 
'''


from turtle import color
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
from src.models.train_DRRAA_module import DRRAA
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.calcNMI import calcNMI
import matplotlib as mpl
import scipy.stats as st

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


iter = 10000
avgNMIs = {}
avgAUCs = {}
conf_NMIs = {}
conf_AUCs = {}
#Get synthetic data and convert to edge list
true_k = 8
true_alpha = 0.2
adj_m, z, A, Z_true, beta = main(alpha=true_alpha, k=true_k, dim=2, nsamples=100, rand=False) #z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())
temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)): 
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#edge_list = torch.FloatTensor(edge_list).long()
num_arc =  [2,3,4,5,6,7,8,9,10]
d = 2
num_init = 5
##Ideal prediction:
Iaucs = []
for _ in range(num_init):
    ideal_score, _, _ = ideal_prediction(adj_m, A, Z_true, beta=beta, test_size = 0.5)
    Iaucs.append(ideal_score)

for k in num_arc:
    NMIs = []
    AUCs = []
    for i in range(num_init):
        model = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    link_pred=True)

        model_nmi = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list)

        model.train(iterations=iter, LR=0.01)
        model_nmi.train(iterations=iter, LR=0.01)
        auc_score, fpr, tpr = model.link_prediction()
        AUCs.append(auc_score)
        Z = F.softmax(model_nmi.Z, dim=0)
        G = F.sigmoid(model_nmi.Gate)
        C = (Z.T * G) / (Z.T * G).sum(0)

        u, sigma, v = torch.svd(model_nmi.A) # Decomposition of A.
        r = torch.matmul(torch.diag(sigma), v.T)
        embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
        archetypes = torch.matmul(r, torch.matmul(Z, C))

        #Calculate NMI between embeddings
        NMIs.append(calcNMI(Z, Z_true).item())
        print(f'The NMI between z and z_hat is {calcNMI(Z, Z_true)}')
    avgNMIs[k] = np.mean(NMIs)
    conf_NMIs[k] =  st.t.interval(alpha=0.95, df=len(NMIs)-1, 
                    loc=np.mean(NMIs), 
                    scale=st.sem(NMIs)) 
    avgAUCs[k] = np.mean(AUCs)
    conf_AUCs[k] = st.t.interval(alpha=0.95, df=len(AUCs)-1, 
                    loc=np.mean(AUCs), 
                    scale=st.sem(AUCs))


fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(num_arc, list(avgNMIs.values()), '-o', label="mean NMI", c="#e3427d")
ax.fill_between(num_arc,
                 y1 = [x for (x,y) in conf_NMIs.values()],
                 y2 = [y for (x,y) in conf_NMIs.values()],
                 color='#e3427d', alpha=0.2)

ax.axvline(true_k, linestyle = '--', color='#303638', label="True number of Archetypes", alpha=0.5)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("NMI")
ax.legend()
ax.grid(alpha=.3)
plt.savefig("complex_convex_NMI.png", dpi=500)
plt.show()

fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(num_arc, list(avgAUCs.values()), '-o', label="RAA", c="#e3427d")
ax.fill_between(num_arc,
                 y1 = [x for (x,y) in conf_AUCs.values()],
                 y2 = [y for (x,y) in conf_AUCs.values()],
                 color="#e3427d", alpha=0.2)
ax.axvline(8, linestyle = '--', color='#303638', label="True number of Archetypes", alpha=0.5)

conf_Iaucs = st.t.interval(alpha=0.95, df=len(Iaucs)-1, 
                        loc=np.mean(Iaucs), 
                        scale=st.sem(Iaucs))

ax.plot(true_k, np.mean(Iaucs),'o',markersize=5, c="#a0c4ff")
ax.errorbar(true_k, np.mean(Iaucs), 
            [abs(x-y)/2 for (x,y) in [conf_Iaucs]],
            solid_capstyle='projecting', capsize=5,
            label="ideal predictor", color='#a0c4ff')

ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("AUC")
ax.grid(alpha=.3)
ax.legend()
plt.savefig("complex_convex.png",dpi=500)
plt.show()

