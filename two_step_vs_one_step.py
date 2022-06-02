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
num_init = 10

raa_best_in_seed_aucs = np.zeros((len(kvals),num_init))
lsmaa_best_in_seed_aucs = np.zeros((len(kvals),num_init))
lsm_best_in_seed_aucs = np.zeros((len(kvals),num_init))
kaa_best_in_seed_aucs = np.zeros((len(kvals),num_init))
raainit_best_in_seed_aucs = np.zeros((len(kvals),num_init))

raa_best_in_seed_nmis = np.zeros((len(kvals),num_init))
lsmaa_best_in_seed_nmis = np.zeros((len(kvals),num_init))
kaa_best_in_seed_nmis = np.zeros((len(kvals),num_init))
raainit_best_in_seed_nmis = np.zeros((len(kvals),num_init))

seed_init = 0

#get ideal prediction:
#ideal_score, _, _ = ideal_prediction(adj_m, G, A, Z_true, beta=beta, test_size=0.3, seed_split=seed_split)

iter = 10000


#################################
## Synthetic model comparison  ##
##       RAA and LSM+AA        ##
#################################
#Defining models



for kval_idx, kval in enumerate(kvals):
    raa_models = []
    lsmaa_models = []
    lsm_models = []
    kaa_models = []
    raainit_models = []

    raa_nmi_models = []
    lsmaa_nmi_models = []
    kaa_nmi_models = []
    raainit_nmi_models = []
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

        lsm = LSM(d=d,
                  sample_size=1,
                  data=edge_list,
                  data_type="edge list",
                  link_pred=True,
                  seed_split=seed_split,
                  seed_init=seed_init
                  )
        lsm.train(iterations=iter)
        lsm_models.append(lsm)

        lsmaa = LSMAA(d=d,
                      k=kval,
                      sample_size=1,
                      data = edge_list,
                      data_type = "Edge list",
                      link_pred = True,
                      seed_split=seed_split,
                      seed_init=seed_init
                      )
        lsmaa.train(iterations=iter)
        lsmaa_models.append(lsmaa)

        kaa = KAA(k=kval,
                  data=edge_list,
                  type='jaccard',
                  data_type='edge list',
                  link_pred=True,
                  seed_split = seed_split,
                  seed_init=seed_init
                  )
        kaa.train(iterations=iter)
        kaa_models.append(kaa)

        kaainit = KAA(k=kval,
                      data=edge_list,
                      data_type="edge list",
                      link_pred=True,
                      seed_split=seed_split,
                      seed_init=seed_init
                      )
        kaainit.train(iterations=1000)

        raainit = DRRAA(init_Z=kaainit.S.detach(),
                        k=kval,
                        d=d,
                        sample_size=1,
                        data=edge_list,
                        data_type="edge list",
                        link_pred=True,
                        seed_split=seed_split,
                        seed_init=seed_init
                        )
        raainit.train(iterations=iter)
        raainit_models.append(raainit)

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

        lsmaa_nmi = LSMAA(d=d,
                      k=kval,
                      sample_size=1,
                      data=edge_list,
                      data_type="Edge list",
                      link_pred=False,
                      seed_init=seed_init
                      )
        lsmaa_nmi.train(iterations=iter)
        lsmaa_nmi_models.append(lsmaa_nmi)

        kaa_nmi = KAA(k=kval,
                  data=adj_m.numpy(),
                  type='jaccard',
                  link_pred=False,
                  seed_init=seed_init
                  )
        kaa_nmi.train(iterations=iter)
        kaa_nmi_models.append(kaa_nmi)

        kaainit = KAA(k=kval,
                      data=adj_m.numpy(),
                      link_pred=False,
                      seed_init=seed_init
                      )
        kaainit.train(iterations=1000)

        raainit_nmi = DRRAA(init_Z=kaainit.S.detach(),
                            k=kval,
                            d=d,
                            sample_size=1,
                            data=edge_list,
                            data_type="edge list",
                            link_pred=False,
                            seed_init=seed_init
                            )
        raainit_nmi.train(iterations=iter)
        raainit_nmi_models.append(raainit_nmi)

        #make sure to increase the initialisation-seed ;)
        seed_init += 1
        print(seed_init)

        raa_best_in_seed_aucs[kval_idx,init], _, _ = raa_models[init].link_prediction()
        lsm_best_in_seed_aucs[kval_idx,init], _, _ = lsm_models[init].link_prediction()
        lsmaa_best_in_seed_aucs[kval_idx,init], _, _ = lsmaa_models[init].link_prediction()
        kaa_best_in_seed_aucs[kval_idx,init], _, _ = kaa_models[init].link_prediction()
        raainit_best_in_seed_aucs[kval_idx, init], _, _ = raainit_models[init].link_prediction()

    for i in range(num_init):
        raa_nmis = []
        lsmaa_nmis = []
        kaa_nmis = []
        raainit_nmis = []
        for j in range(num_init):
            if i != j:
                raa_nmis.append(calcNMI(F.softmax(raa_nmi_models[i].Z.detach(), dim=0), F.softmax(raa_nmi_models[j].Z.detach(), dim=0)).float().item())

                aa = arch.AA(n_archetypes=kval)
                Zi = aa.fit_transform(lsmaa_nmi_models[i].latent_Z.detach().numpy())
                Zj = aa.fit_transform(lsmaa_nmi_models[j].latent_Z.detach().numpy())
                lsmaa_nmis.append(calcNMI(torch.from_numpy(Zi).T, torch.from_numpy(Zj).T).float().item())

                kaa_nmis.append(calcNMI(F.softmax(kaa_nmi_models[i].S.detach(), dim=0), F.softmax(kaa_nmi_models[j].S.detach(), dim=0)).float().item())

                raainit_nmis.append(calcNMI(F.softmax(raainit_nmi_models[i].Z.detach(), dim=0), F.softmax(raainit_nmi_models[j].Z.detach(), dim=0)).float().item())
        raa_best_in_seed_nmis[kval_idx, i] = np.mean(raa_nmis)
        lsmaa_best_in_seed_nmis[kval_idx, i] = np.mean(lsmaa_nmis)
        kaa_best_in_seed_nmis[kval_idx, i] = np.mean(kaa_nmis)
        raainit_best_in_seed_nmis[kval_idx, i] = np.mean(raainit_nmis)

avg_raa_aucs = np.mean(raa_best_in_seed_aucs, 1)
avg_lsmaa_aucs = np.mean(lsmaa_best_in_seed_aucs, 1)
avg_lsm_aucs = np.mean(lsm_best_in_seed_aucs, 1)
avg_kaa_aucs = np.mean(kaa_best_in_seed_aucs, 1)
avg_raainit_aucs = np.mean(raainit_best_in_seed_aucs, 1)

conf_raa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_aucs,
                        scale=st.sem(raa_best_in_seed_aucs, 1))
conf_lsmaa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_lsmaa_aucs,
                        scale=st.sem(lsmaa_best_in_seed_aucs, 1))
conf_lsm_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                       loc=avg_lsm_aucs,
                        scale=st.sem(lsm_best_in_seed_aucs, 1))
conf_kaa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_aucs,
                        scale=st.sem(kaa_best_in_seed_aucs, 1))
conf_raainit_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raainit_aucs,
                        scale=st.sem(raainit_best_in_seed_aucs, 1))

avg_raa_nmis = np.mean(raa_best_in_seed_nmis,1)
avg_lsmaa_nmis = np.mean(lsmaa_best_in_seed_nmis,1)
avg_kaa_nmis = np.mean(kaa_best_in_seed_nmis,1)
avg_raainit_nmis = np.mean(raainit_best_in_seed_nmis,1)

conf_raa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_nmis,
                        scale=st.sem(raa_best_in_seed_nmis, 1))
conf_lsmaa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_lsmaa_nmis,
                        scale=st.sem(lsmaa_best_in_seed_nmis, 1))
conf_kaa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_nmis,
                        scale=st.sem(kaa_best_in_seed_nmis, 1))
conf_raainit_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raainit_nmis,
                        scale=st.sem(raainit_best_in_seed_nmis, 1))

#AUC plot
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(kvals, avg_raa_aucs, '-o', label="RAA", color='#e3427d')
ax.fill_between(kvals,
                 y1 = conf_raa_aucs[0],
                 y2 = conf_raa_aucs[1],
                 color='#e3427d', alpha=0.2)
ax.plot(kvals, conf_raa_aucs[0], '--', color='#e3427d')
ax.plot(kvals, conf_raa_aucs[1], '--', color='#e3427d')

ax.plot(kvals, avg_lsmaa_aucs, '-o', label="LSM+AA", color='#bdb2ff')
ax.fill_between(kvals,
                 y1 = conf_lsmaa_aucs[0],
                 y2 = conf_lsmaa_aucs[1],
                 color='#bdb2ff', alpha=0.2)
ax.plot(kvals, conf_lsmaa_aucs[0], '--', color='#bdb2ff')
ax.plot(kvals, conf_lsmaa_aucs[1], '--', color='#bdb2ff')

ax.plot(kvals, avg_lsm_aucs, '-o', label="LSM", color='#e68653')
ax.fill_between(kvals,
                 y1 = conf_lsm_aucs[0],
                 y2 = conf_lsm_aucs[1],
                 color='#e68653', alpha=0.2)
ax.plot(kvals, conf_lsm_aucs[0], '--', color='#e68653')
ax.plot(kvals, conf_lsm_aucs[1], '--', color='#e68653')

ax.plot(kvals, avg_kaa_aucs, '-o', label="KAA", color='#ffd6a5')
ax.fill_between(kvals,
                 y1 = conf_kaa_aucs[0],
                 y2 = conf_kaa_aucs[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(kvals, conf_kaa_aucs[0], '--', color='#ffd6a5')
ax.plot(kvals, conf_kaa_aucs[1], '--', color='#ffd6a5')

ax.plot(kvals, avg_raainit_aucs, '-o', label="RAA with init", color='#746ab0')
ax.fill_between(kvals,
                 y1 = conf_raainit_aucs[0],
                 y2 = conf_raainit_aucs[1],
                 color='#746ab0', alpha=0.2)
ax.plot(kvals, conf_raainit_aucs[0], '--', color='#746ab0')
ax.plot(kvals, conf_raainit_aucs[1], '--', color='#746ab0')

ax.axvline(K, linestyle = '--', color='#303638', label="True number of Archetypes", alpha=0.5)
ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("AUC")
ax.legend()
plt.savefig('two_step_vs_one_step_auc.png',dpi=500)
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

ax.plot(kvals, avg_lsmaa_nmis, '-o', label="LSM+AA", color='#bdb2ff')
ax.fill_between(kvals,
                 y1 = conf_lsmaa_nmis[0],
                 y2 = conf_lsmaa_nmis[1],
                 color='#bdb2ff', alpha=0.2)
ax.plot(kvals, conf_lsmaa_nmis[0], '--', color='#bdb2ff')
ax.plot(kvals, conf_lsmaa_nmis[1], '--', color='#bdb2ff')

ax.plot(kvals, avg_kaa_nmis, '-o', label="KAA", color='#ffd6a5')
ax.fill_between(kvals,
                 y1 = conf_kaa_nmis[0],
                 y2 = conf_kaa_nmis[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(kvals, conf_kaa_nmis[0], '--', color='#ffd6a5')
ax.plot(kvals, conf_kaa_nmis[1], '--', color='#ffd6a5')

ax.plot(kvals, avg_raainit_nmis, '-o', label="RAA with init", color='#746ab0')
ax.fill_between(kvals,
                 y1 = conf_raainit_nmis[0],
                 y2 = conf_raainit_nmis[1],
                 color='#746ab0', alpha=0.2)
ax.plot(kvals, conf_raainit_nmis[0], '--', color='#746ab0')
ax.plot(kvals, conf_raainit_nmis[1], '--', color='#746ab0')

ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("NMI")
ax.legend()
plt.savefig('two_step_vs_one_step_nmi.png',dpi=500)
#plt.show()