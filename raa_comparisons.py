'''
LDM with AA on embeddings vs. RAA:
		Run with synthetic data K = 3
		Synthetic alphas = 0.2
		N = 100 (wip)
		K = 2 .. 10
		D = 2
		Inits = 5           #Number of inits.
		seed = 1999
		sample_size = 1
		Lr = 0.01
		Iterations = 10,000
'''
from src.models.train_DRRAA_module import DRRAA
from src.models.train_DRRAA_ngating import DRRAA_ngating
from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_DRRAA_bare import DRRAA_bare
from src.models.calcNMI import calcNMI
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx
import archetypes as arch
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

rand = False
if rand:
    np.random.seed(1)
    torch.manual_seed(1)
else:
    np.random.seed(42)
    torch.manual_seed(42)

#create data before the runs to make sure we test initialisations of models:
real_alpha = 0.2
K = 3
n = 100
d = 2
adj_m, z, A, Z_true, beta, partition = main(alpha=real_alpha, k=K, dim=d, nsamples=n, rand=rand)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#set test and train split seed. We want the same train and test split in order to know that differences
#are because of inits.
seed_split = 42

#Run 10 different seeds with 10 different inits. Take the best of the 10 inits and save as best in seed.
#Then plot the auc and nmi with errorbars on the 10 best in seeds.
kvals = [2,3,4,5,6]
num_init = 10000

raa_best_in_seed_aucs = np.zeros((len(kvals),num_init))
raa_ng_best_in_seed_aucs = np.zeros((len(kvals),num_init))
raa_nre_best_in_seed_aucs = np.zeros((len(kvals),num_init))
raa_bare_best_in_seed_aucs = np.zeros((len(kvals),num_init))

raa_best_in_seed_nmis = np.zeros((len(kvals),num_init))
raa_ng_best_in_seed_nmis = np.zeros((len(kvals),num_init))
raa_nre_best_in_seed_nmis = np.zeros((len(kvals),num_init))
raa_bare_best_in_seed_nmis = np.zeros((len(kvals),num_init))

seed_init = 0

#get ideal prediction:
#ideal_score, _, _ = ideal_prediction(adj_m, G, A, Z_true, beta=beta, test_size=0.3, seed_split=seed_split)

iter = 10


#################################
## Synthetic model comparison  ##
##       RAA and LSM+AA        ##
#################################
#Defining models



for kval_idx, kval in enumerate(kvals):
    raa_models = []
    raa_ng_models = []
    raa_nre_models = []
    raa_bare_models = []

    raa_nmi_models = []
    raa_ng_nmi_models = []
    raa_nre_nmi_models = []
    raa_bare_nmi_models = []
    for init in range(num_init):
        raa = DRRAA(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = True,
                    seed_split=seed_split,
                    seed_init=seed_init
        )
        raa.train(iterations=iter)
        raa_models.append(raa)

        raa_ng = DRRAA_ngating(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = True,
                    seed_split=seed_split,
                    seed_init=seed_init
        )
        raa_ng.train(iterations=iter)
        raa_ng_models.append(raa_ng)

        raa_nre = DRRAA_nre(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = True,
                    seed_split=seed_split,
                    seed_init=seed_init
        )
        raa_nre.train(iterations=iter)
        raa_nre_models.append(raa_nre)

        raa_bare = DRRAA_bare(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = True,
                    seed_split=seed_split,
                    seed_init=seed_init
        )
        raa_bare.train(iterations=iter)
        raa_bare_models.append(raa_bare)

        #train on whole data set for nmis ;)
        raa_nmi = DRRAA(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type="edge list",
                    link_pred=False,
                    seed_init=seed_init
                    )
        raa_nmi.train(iterations=iter)
        raa_nmi_models.append(raa_nmi)

        raa_ng_nmi = DRRAA_ngating(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type="edge list",
                    link_pred=False,
                    seed_init=seed_init
                    )
        raa_ng_nmi.train(iterations=iter)
        raa_ng_nmi_models.append(raa_ng_nmi)

        raa_nre_nmi = DRRAA_nre(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type="edge list",
                    link_pred=False,
                    seed_init=seed_init
                    )
        raa_nre_nmi.train(iterations=iter)
        raa_nre_nmi_models.append(raa_nre_nmi)

        raa_bare_nmi = DRRAA_bare(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type="edge list",
                    link_pred=False,
                    seed_init=seed_init
                    )
        raa_bare_nmi.train(iterations=iter)
        raa_bare_nmi_models.append(raa_bare_nmi)

        #make sure to increase the initialisation-seed ;)
        seed_init += 1
        print(seed_init)

        raa_best_in_seed_aucs[kval_idx,init], _, _ = raa_models[init].link_prediction()
        raa_ng_best_in_seed_aucs[kval_idx,init], _, _ = raa_ng_models[init].link_prediction()
        raa_nre_best_in_seed_aucs[kval_idx,init], _, _ = raa_nre_models[init].link_prediction()
        raa_bare_best_in_seed_aucs[kval_idx, init], _, _ = raa_bare_models[init].link_prediction()

    for i in range(num_init):
        raa_nmis = []
        raa_ng_nmis = []
        raa_nre_nmis = []
        raa_bare_nmis = []
        for j in range(num_init):
            if i != j:
                raa_nmis.append(calcNMI(F.softmax(raa_nmi_models[i].Z.detach(), dim=0), F.softmax(raa_nmi_models[j].Z.detach(), dim=0)).float().item())
                raa_ng_nmis.append(calcNMI(F.softmax(raa_ng_nmi_models[i].Z.detach(), dim=0),
                                        F.softmax(raa_ng_nmi_models[j].Z.detach(), dim=0)).float().item())
                raa_nre_nmis.append(calcNMI(F.softmax(raa_nre_nmi_models[i].Z.detach(), dim=0),
                                        F.softmax(raa_nre_nmi_models[j].Z.detach(), dim=0)).float().item())
                raa_bare_nmis.append(calcNMI(F.softmax(raa_bare_nmi_models[i].Z.detach(), dim=0),
                                        F.softmax(raa_bare_nmi_models[j].Z.detach(), dim=0)).float().item())

        raa_best_in_seed_nmis[kval_idx, i] = np.mean(raa_nmis)
        raa_ng_best_in_seed_nmis[kval_idx, i] = np.mean(raa_ng_nmis)
        raa_nre_best_in_seed_nmis[kval_idx, i] = np.mean(raa_nre_nmis)
        raa_bare_best_in_seed_nmis[kval_idx, i] = np.mean(raa_bare_nmis)

avg_raa_aucs = np.mean(raa_best_in_seed_aucs, 1)
avg_raa_ng_aucs = np.mean(raa_ng_best_in_seed_aucs, 1)
avg_raa_nre_aucs = np.mean(raa_nre_best_in_seed_aucs, 1)
avg_raa_bare_aucs = np.mean(raa_bare_best_in_seed_aucs, 1)

conf_raa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_aucs,
                        scale=st.sem(raa_best_in_seed_aucs, 1))
conf_raa_ng_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_ng_aucs,
                        scale=st.sem(raa_ng_best_in_seed_aucs, 1))
conf_raa_nre_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_nre_aucs,
                        scale=st.sem(raa_nre_best_in_seed_aucs, 1))
conf_raa_bare_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_bare_aucs,
                        scale=st.sem(raa_bare_best_in_seed_aucs, 1))

avg_raa_nmis = np.mean(raa_best_in_seed_nmis,1)
avg_raa_ng_nmis = np.mean(raa_ng_best_in_seed_nmis,1)
avg_raa_nre_nmis = np.mean(raa_nre_best_in_seed_nmis,1)
avg_raa_bare_nmis = np.mean(raa_bare_best_in_seed_nmis,1)

conf_raa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_nmis,
                        scale=st.sem(raa_best_in_seed_nmis, 1))
conf_raa_ng_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_ng_nmis,
                        scale=st.sem(raa_ng_best_in_seed_nmis, 1))
conf_raa_nre_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_nre_nmis,
                        scale=st.sem(raa_nre_best_in_seed_nmis, 1))
conf_raa_bare_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_bare_nmis,
                        scale=st.sem(raa_bare_best_in_seed_nmis, 1))

#AUC plot
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(kvals, avg_raa_aucs, '-o', label="RAA", color='#e3427d')
ax.fill_between(kvals,
                 y1 = conf_raa_aucs[0],
                 y2 = conf_raa_aucs[1],
                 color='#e3427d', alpha=0.2)
ax.plot(kvals, conf_raa_aucs[0], '--', color='#e3427d')
ax.plot(kvals, conf_raa_aucs[1], '--', color='#e3427d')

ax.plot(kvals, avg_raa_ng_aucs, '-o', label="RAA no gating", color='#4c6e81')
ax.fill_between(kvals,
                 y1 = conf_raa_ng_aucs[0],
                 y2 = conf_raa_ng_aucs[1],
                 color='#4c6e81', alpha=0.2)
ax.plot(kvals, conf_raa_ng_aucs[0], '--', color='#4c6e81')
ax.plot(kvals, conf_raa_ng_aucs[1], '--', color='#4c6e81')

ax.plot(kvals, avg_raa_nre_aucs, '-o', label="RAA no random effects", color='#5d4b20')
ax.fill_between(kvals,
                 y1 = conf_raa_nre_aucs[0],
                 y2 = conf_raa_nre_aucs[1],
                 color='#5d4b20', alpha=0.2)
ax.plot(kvals, conf_raa_nre_aucs[0], '--', color='#5d4b20')
ax.plot(kvals, conf_raa_nre_aucs[1], '--', color='#5d4b20')

ax.plot(kvals, avg_raa_bare_aucs, '-o', label="RAA no gating or random effects", color='#f0c808')
ax.fill_between(kvals,
                 y1 = conf_raa_bare_aucs[0],
                 y2 = conf_raa_bare_aucs[1],
                 color='#f0c808', alpha=0.2)
ax.plot(kvals, conf_raa_bare_aucs[0], '--', color='#f0c808')
ax.plot(kvals, conf_raa_bare_aucs[1], '--', color='#f0c808')

ax.axvline(K, linestyle = '--', color='#303638', label="True number of Archetypes", alpha=0.5)
ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("AUC")
ax.legend()
plt.savefig('raa_comparisons_auc.png',dpi=500)
#plt.show()


#NMI plot:
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(kvals, avg_raa_nmis, '-o', label="RAA", color='#e3427d')
ax.fill_between(kvals,
                 y1 = conf_raa_nmis[0],
                 y2 = conf_raa_nmis[1],
                 color='#e3427d', alpha=0.2)
ax.plot(kvals, conf_raa_nmis[0], '--', color='#e3427d')
ax.plot(kvals, conf_raa_nmis[1], '--', color='#e3427d')

ax.plot(kvals, avg_raa_ng_nmis, '-o', label="RAA no gating", color='#4c6e81')
ax.fill_between(kvals,
                 y1 = conf_raa_ng_nmis[0],
                 y2 = conf_raa_ng_nmis[1],
                 color='#4c6e81', alpha=0.2)
ax.plot(kvals, conf_raa_ng_nmis[0], '--', color='#4c6e81')
ax.plot(kvals, conf_raa_ng_nmis[1], '--', color='#4c6e81')

ax.plot(kvals, avg_raa_nre_nmis, '-o', label="RAA no random effects", color='#5d4b20')
ax.fill_between(kvals,
                 y1 = conf_raa_nre_nmis[0],
                 y2 = conf_raa_nre_nmis[1],
                 color='#5d4b20', alpha=0.2)
ax.plot(kvals, conf_raa_nre_nmis[0], '--', color='#5d4b20')
ax.plot(kvals, conf_raa_nre_nmis[1], '--', color='#5d4b20')

ax.plot(kvals, avg_raa_bare_nmis, '-o', label="RAA no gating or random effects", color='#f0c808')
ax.fill_between(kvals,
                 y1 = conf_raa_bare_nmis[0],
                 y2 = conf_raa_bare_nmis[1],
                 color='#f0c808', alpha=0.2)
ax.plot(kvals, conf_raa_bare_nmis[0], '--', color='#f0c808')
ax.plot(kvals, conf_raa_bare_nmis[1], '--', color='#f0c808')

ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("NMI")
ax.legend()
plt.savefig('raa_comparisons_nmi.png',dpi=500)
#plt.show()