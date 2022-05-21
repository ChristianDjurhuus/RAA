'''
Compare the three models (KAA, RAA, LDM): 
    Ideal predictor: True 
    Run with synthetic data K = 8 
    Synthetic alphas = 0.2, 1, 5 
    N = 100 (wip) 
    K = 2 .. 10 
    D = 2 
    CV = 5 
    seed = 42
    sample_size = 1 
    Lr = 0.01 
    Iterations = 10,000 
'''

from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
from src.models.train_KAA_module import KAA
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl 
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

#################################
## Synthetic model comparison  ##
##      RAA and baselines      ##
#################################

#Data creation
real_alpha = 0.2
K = 3
n = 100
d = 2
adj_m, z, A, Z_true, beta = main(alpha=real_alpha, k=K, dim=d, nsamples=n, rand=False)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)): 
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#Defining models
iter = 10
num_init = 5
kvals = [2,3,4,5,6,7,8]

avg_raa_aucs = {}
avg_kaa_aucs = {}
conf_raa_aucs = {}
conf_kaa_aucs = {}
Iaucs = []
lsm_aucs = []

best_lsm_auc = 0
best_raa_auc = 0

for _ in range(num_init):
    lsm = LSM(d=d, 
                sample_size=1, #Without random sampling
                data=edge_list,
                link_pred=True)

    lsm.train(iterations=iter)
    lsm_auc, _, _ = lsm.link_prediction()
    lsm_aucs.append(lsm_auc)
    if lsm_auc > best_lsm_auc:
        lsm_best = lsm

    #Prediction with ideal embeddings
    ideal_score, _, _ = ideal_prediction(adj_m, A, Z_true, beta=beta, test_size = 0.3)
    Iaucs.append(ideal_score)

for kval in kvals:
    raa_aucs = []
    kaa_aucs = []
    for _ in range(num_init):
        raa = DRRAA(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = True
        )
        raa.train(iterations=iter)
        raa_auc, _, _ = raa.link_prediction()
        raa_aucs.append(raa_auc)
        if raa_auc > best_raa_auc:
            raa_best = raa

        kaa = KAA(k=kval,
                  data=adj_m.numpy(),
                  type="jaccard",
                  link_pred=True,                
        )
        kaa.train(iterations=iter)
        kaa_auc, _, _ = kaa.link_prediction()
        kaa_aucs.append(kaa_auc)

    avg_raa_aucs[kval] = np.mean(raa_aucs)
    avg_kaa_aucs[kval] = np.mean(kaa_aucs)
    conf_raa_aucs[kval] = st.t.interval(alpha=0.95, df=len(raa_aucs)-1, 
                        loc=np.mean(raa_aucs), 
                        scale=st.sem(raa_aucs))
    conf_kaa_aucs[kval] = st.t.interval(alpha=0.95, df=len(kaa_aucs)-1, 
                        loc=np.mean(kaa_aucs), 
                        scale=st.sem(kaa_aucs))


fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(kvals, avg_raa_aucs.values(), '-o', label="RAA", color='#e3427d')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in conf_raa_aucs.values()],
                 y2 = [y for (x,y) in conf_raa_aucs.values()],
                 color='#e3427d', alpha=0.2)
ax.plot(kvals, [x for (x,y) in conf_raa_aucs.values()], '--', color='#e3427d')
ax.plot(kvals, [y for (x,y) in conf_raa_aucs.values()], '--', color='#e3427d')

ax.plot(kvals, avg_kaa_aucs.values(), '-o', label="KAA (Jaccard)", color='#ffd6a5')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in conf_kaa_aucs.values()],
                 y2 = [y for (x,y) in conf_kaa_aucs.values()],
                 color='#ffd6a5', alpha=0.2)
ax.plot(kvals, [x for (x,y) in conf_kaa_aucs.values()], '--', color='#ffd6a5')
ax.plot(kvals, [y for (x,y) in conf_kaa_aucs.values()], '--', color='#ffd6a5')


avg_lsm_aucs = [0]*len(kvals) + np.mean(lsm_aucs)
conf_lsm_aucs = st.t.interval(alpha=0.95, df=len(lsm_aucs)-1, 
                        loc=np.mean(lsm_aucs), 
                        scale=st.sem(lsm_aucs))

ax.plot(kvals, avg_lsm_aucs, '-o', label="LDM", color="#e68653")
ax.fill_between(kvals,
                 y1 = [0]*len(kvals) + conf_lsm_aucs[0],
                 y2 = [0]*len(kvals) + conf_lsm_aucs[1],
                 color='#e68653', alpha=0.2)
ax.plot(kvals, [0]*len(kvals) + conf_lsm_aucs[0], '--', color='#e68653')
ax.plot(kvals,  [0]*len(kvals) + conf_lsm_aucs[1], '--', color='#e68653')

conf_Iaucs = st.t.interval(alpha=0.95, df=len(Iaucs)-1, 
                        loc=np.mean(Iaucs), 
                        scale=st.sem(Iaucs))

ax.plot(K,np.mean(Iaucs),'o',markersize=5,color='#a0c4ff')
ax.errorbar(K, np.mean(Iaucs), 
            [abs(x-y)/2 for (x,y) in [conf_Iaucs]],
            solid_capstyle='projecting', capsize=5,
            label="ideal predictor", color='#a0c4ff')

ax.axvline(K, linestyle = '--', color='#303638', label="True number of Archetypes", alpha=0.5)
ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("AUC")
ax.legend()
plt.savefig("synthetic_model_comparison.png", dpi=500)
#plt.show()


raa_best.plot_latent_and_loss(iterations=iter, cmap=z)
kaa.plot_latent_and_loss(iterations=iter, cmap=z)
lsm_best.plot_latent_and_loss(iterations=iter, cmap=z)