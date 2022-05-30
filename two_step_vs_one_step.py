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

np.random.seed(42)
torch.manual_seed(42)

top10 = np.arange(2)

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

#Run 10 different seeds with 10 different inits. Take the best of the 10 inits and save as best in seed.
#Then plot the auc and nmi with errorbars on the 10 best in seeds.
kvals = [2,3,4,5,6,7,8]

raa_best_in_seed_aucs = np.zeros((len(top10),len(kvals)))
lsmaa_best_in_seed_aucs = np.zeros((len(top10),len(kvals)))
lsm_best_in_seed_aucs = np.zeros((len(top10),len(kvals)))
kaa_best_in_seed_aucs = np.zeros((len(top10),len(kvals)))

raa_best_in_seed_nmis = np.zeros((len(top10),len(kvals)))
lsmaa_best_in_seed_nmis = np.zeros((len(top10),len(kvals)))
lsm_best_in_seed_nmis = np.zeros((len(top10),len(kvals)))
kaa_best_in_seed_nmis = np.zeros((len(top10),len(kvals)))

seed_init = 0


#get ideal prediction:
ideal_score, _, _ = ideal_prediction(adj_m, G, A, Z_true, beta=beta, test_size=0.3, seed_split=seed_split)


iter = 10000
num_init = 2
for big_iteration in top10:
    #################################
    ## Synthetic model comparison  ##
    ##       RAA and LSM+AA        ##
    #################################
    #Defining models

    raa_models = {}
    lsmaa_models = {}
    lsm_models = {}
    kaa_models = {}

    raa_nmi_models = {}
    lsmaa_nmi_models = {}
    lsm_nmi_models = {}
    kaa_nmi_models = {}

    best_loss_lsm = 10000
    best_loss_lsm_nmi = 10000
    for init in range(num_init):
        lsm = LSM(d=d,
                    sample_size=1,
                    data = edge_list,
                    data_type="edge list",
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init
                    )
        lsm.train(iterations=iter)
        if np.mean(lsm.losses[-100:]) < best_loss_lsm:
            lsm_models[big_iteration] = lsm
            best_loss_lsm = np.mean(lsm.losses[-100:])

    


    for kval in kvals:
        best_loss_raa = 10000
        best_loss_lsmaa = 10000
        best_loss_kaa = 10000

        best_loss_raa_nmi = 10000
        best_loss_lsmaa_nmi = 10000
        best_loss_kaa_nmi = 10000
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
            if np.mean(raa.losses[-100:]) < best_loss_raa:
                raa_models[kval] = raa #Should they be updated here?
                best_loss_raa = np.mean(raa.losses[-100:])

            lsmaa = LSMAA(d=d+1,
                          k=kval,
                          sample_size=1,
                          data = edge_list,
                          data_type = "Edge list",
                          link_pred = True,
                          seed_split=seed_split,
                          seed_init=seed_init
                          )
            lsmaa.train(iterations=iter)
            if np.mean(lsmaa.losses[-100:]) < best_loss_lsmaa:
                lsmaa_models[kval] = lsmaa
                best_loss_lsmaa = np.mean(lsmaa.losses[-100:])

            kaa = KAA(k=kval,
                      data=adj_m.numpy(),
                      type='jaccard',
                      link_pred=True
                      )
            kaa.train(iterations=iter)
            if np.mean(kaa.losses[-100:]) < best_loss_kaa:
                kaa_models[kval] = kaa
                best_loss_kaa = np.mean(kaa.losses[-100:])

            #############################################################################
            #NMIs - require full data, so link_pred=False, else everything is the same :)
            raa_nmi = DRRAA(k=kval,
                        d=d,
                        sample_size=1,
                        data=edge_list,
                        data_type="edge list",
                        link_pred=False,
                        seed_init=seed_init
                        )
            raa_nmi.train(iterations=iter)
            if np.mean(raa_nmi.losses[-100:]) < best_loss_raa_nmi:
                raa_nmi_models[kval] = raa_nmi

            lsmaa_nmi = LSMAA(d=d,
                          k=kval,
                          sample_size=1,
                          data=edge_list,
                          data_type="Edge list",
                          link_pred=False,
                          seed_init=seed_init
                          )
            lsmaa_nmi.train(iterations=iter)
            if np.mean(lsmaa_nmi.losses[-100:]) < best_loss_lsmaa_nmi:
                lsmaa_nmi_models[kval] = lsmaa_nmi


            kaa_nmi = KAA(k=kval,
                      data=adj_m.numpy(),
                      type='jaccard',
                      link_pred=False
                      )
            kaa_nmi.train(iterations=iter)
            if np.mean(kaa_nmi.losses[-100:]) < best_loss_kaa_nmi:
                kaa_nmi_models[kval] = kaa_nmi

            #make sure to increase the initialisation-seed ;)
            seed_init += 1
            print(seed_init)

    raa_aucs = []
    lsmaa_aucs = []
    lsm_aucs = []
    kaa_aucs = []

    raa_nmis = []
    lsmaa_nmis = []
    lsm_nmis = []
    kaa_nmis = []
    
    
    lsm_auc, _, _ = lsm_models[big_iteration].link_prediction()
    Z_lsm = F.softmax(lsm_models[big_iteration].latent_Z, dim=0)
    lsm_nmi = calcNMI(Z_lsm.detach().T, Z_true) #maybe detach().numpy()
    lsm_aucs.append(lsm_auc)
    lsm_nmis.append(lsm_nmi)

    for key in raa_models.keys():
        #calc aucs
        raa_auc, _, _ = raa_models[key].link_prediction()
        lsmaa_auc, _, _ = lsmaa_models[key].link_prediction()
        #lsm_auc, _, _ = lsm_models[key].link_prediction()
        kaa_auc, _, _ = kaa_models[key].link_prediction()

        raa_aucs.append(raa_auc)
        lsmaa_aucs.append(lsmaa_auc)
        #lsm_aucs.append(lsm_auc)
        kaa_aucs.append(kaa_auc)

        #calc nmis
        raa_nmi = calcNMI(raa_nmi_models[key].Z.detach(), Z_true)
        aa = arch.AA(n_archetypes=key)
        Z = aa.fit_transform(lsmaa.latent_Z.detach().numpy())
        lsmaa_nmi = calcNMI(torch.from_numpy(Z).T.float(), Z_true)
        #lsm_nmi = calcNMI(lsm_nmi_models[key].latent_Z.detach().T, Z_true)
        kaa_nmi = calcNMI(kaa_nmi_models[key].S.detach(), Z_true)

        raa_nmis.append(raa_nmi)
        lsmaa_nmis.append(lsmaa_nmi)
        #lsm_nmis.append(lsm_nmi)
        kaa_nmis.append(kaa_nmi)

    #append aucs and NMIs
    raa_best_in_seed_aucs[big_iteration,:] = raa_aucs
    lsmaa_best_in_seed_aucs[big_iteration,:] = lsmaa_aucs
    lsm_best_in_seed_aucs[big_iteration,:] = lsm_aucs
    kaa_best_in_seed_aucs[big_iteration,:] = kaa_aucs

    raa_best_in_seed_nmis[big_iteration,:] = raa_nmis
    lsmaa_best_in_seed_nmis[big_iteration,:] = lsmaa_nmis
    lsm_best_in_seed_nmis[big_iteration,:] = lsm_nmis
    kaa_best_in_seed_nmis[big_iteration,:] = kaa_nmis

avg_raa_aucs = np.mean(raa_best_in_seed_aucs,0)
avg_lsmaa_aucs = np.mean(lsmaa_best_in_seed_aucs,0)
avg_lsm_aucs = np.mean(lsm_best_in_seed_aucs,0)
avg_kaa_aucs = np.mean(kaa_best_in_seed_aucs,0)

conf_raa_aucs = st.t.interval(alpha=0.95, df=len(avg_raa_aucs)-1,
                        loc=avg_raa_aucs,
                        scale=st.sem(raa_best_in_seed_aucs))
conf_lsmaa_aucs = st.t.interval(alpha=0.95, df=len(avg_raa_aucs)-1,
                        loc=avg_lsmaa_aucs,
                        scale=st.sem(lsmaa_best_in_seed_aucs))
conf_lsm_aucs = st.t.interval(alpha=0.95, df=len(avg_raa_aucs)-1,
                       loc=avg_lsm_aucs,
                        scale=st.sem(lsm_best_in_seed_aucs))
conf_kaa_aucs = st.t.interval(alpha=0.95, df=len(avg_raa_aucs)-1,
                        loc=avg_kaa_aucs,
                        scale=st.sem(kaa_best_in_seed_aucs))

avg_raa_nmis = np.mean(raa_best_in_seed_nmis,0)
avg_lsmaa_nmis = np.mean(lsmaa_best_in_seed_nmis,0)
avg_lsm_nmis = np.mean(lsm_best_in_seed_nmis,0)
avg_kaa_nmis = np.mean(kaa_best_in_seed_nmis,0)

conf_raa_nmis = st.t.interval(alpha=0.95, df=len(avg_raa_nmis)-1,
                        loc=avg_raa_nmis,
                        scale=st.sem(raa_best_in_seed_nmis))
conf_lsmaa_nmis = st.t.interval(alpha=0.95, df=len(avg_raa_nmis)-1,
                        loc=avg_lsmaa_nmis,
                        scale=st.sem(lsmaa_best_in_seed_nmis))
conf_lsm_nmis = st.t.interval(alpha=0.95, df=len(avg_raa_nmis)-1,
                        loc=avg_lsm_nmis,
                        scale=st.sem(lsm_best_in_seed_nmis))
conf_kaa_nmis = st.t.interval(alpha=0.95, df=len(avg_raa_nmis)-1,
                        loc=avg_kaa_nmis,
                        scale=st.sem(kaa_best_in_seed_nmis))

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

ax.plot(K,ideal_score,'o', markersize=5, color='#a0c4ff', label="Ideal Predicter")
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

ax.plot(kvals, avg_lsm_nmis, '-o', label="LSM", color='#e68653')
ax.fill_between(kvals,
                 y1 = conf_lsm_nmis[0],
                 y2 = conf_lsm_nmis[1],
                 color='#e68653', alpha=0.2)
ax.plot(kvals, conf_lsm_nmis[0], '--', color='#e68653')
ax.plot(kvals, conf_lsm_nmis[1], '--', color='#e68653')

ax.plot(kvals, avg_kaa_nmis, '-o', label="KAA", color='#ffd6a5')
ax.fill_between(kvals,
                 y1 = conf_kaa_nmis[0],
                 y2 = conf_kaa_nmis[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(kvals, conf_kaa_nmis[0], '--', color='#ffd6a5')
ax.plot(kvals, conf_kaa_nmis[1], '--', color='#ffd6a5')

ax.axvline(K, linestyle = '--', color='#303638', label="True number of Archetypes", alpha=0.5)
ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("NMI")
ax.legend()
plt.savefig('two_step_vs_one_step_nmi.png',dpi=500)