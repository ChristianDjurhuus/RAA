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
from src.models.train_LSM_module import LSM, LSMAA
from src.models.train_KAA_module import KAA
from src.models.calcNMI import calcNMI

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl 
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx
import archetypes as arch
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

rand=False

#set test and train split seed. We want the same train and test split in order to know that differences
#are because of inits.
seed_split = 42

#Run 10 different seeds with 10 different inits. Take the best of the 10 inits and save as best in seed.
#Then plot the auc and nmi with errorbars on the 10 best in seeds.
alphas = [0.2,1,5]
num_init = 10

raa_best_in_seed_aucs = np.zeros((len(alphas),num_init))
lsm_best_in_seed_aucs = np.zeros((len(alphas),num_init))
lsmaa_best_in_seed_aucs = np.zeros((len(alphas),num_init))
kaa_best_in_seed_aucs = np.zeros((len(alphas),num_init))
raainit_best_in_seed_aucs = np.zeros((len(alphas),num_init))

raa_best_in_seed_nmis = np.zeros((len(alphas),num_init))
lsm_best_in_seed_nmis = np.zeros((len(alphas),num_init))
lsmaa_best_in_seed_nmis = np.zeros((len(alphas),num_init))
kaa_best_in_seed_nmis = np.zeros((len(alphas),num_init))
raainit_best_in_seed_nmis = np.zeros((len(alphas),num_init))

seed_init = 0


####################################
##    Synthetic model comparison  ##
## RAA and all RAAs without stuff ##
####################################
#Defining models
iter = 10000




for alpha_idx, alpha in enumerate(alphas):
    if rand:
        np.random.seed(1)
        torch.manual_seed(1)
    else:
        np.random.seed(42)
        torch.manual_seed(42)

    k=3
    n = 100
    d = 2
    adj_m, z, A, Z_true, beta, partition = main(alpha=alpha, k=k, dim=d, nsamples=n, rand=rand)
    G = nx.from_numpy_matrix(adj_m.numpy())

    temp = [x for x in nx.generate_edgelist(G, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)):
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    raa_models = []
    lsm_models = []
    lsmaa_models = []
    kaa_models = []
    raainit_models = []

    raa_nmi_models = []
    lsm_nmi_models = []
    lsmaa_nmi_models = []
    kaa_nmi_models = []
    raainit_nmi_models = []

    for init in range(num_init):
        raa = DRRAA(k=k,
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
                    seed_init=seed_init,
                    seed_split = seed_split
                    )
        lsm.train(iterations=iter)
        lsm_models.append(lsm)

        lsmaa = LSMAA(d=d,
                      k=k,
                      sample_size=1,
                      data = edge_list,
                      data_type = "Edge list",
                      link_pred = True,
                      seed_split=seed_split,
                      seed_init=seed_init
                      )
        lsmaa.train(iterations=iter)
        lsmaa_models.append(lsmaa)

        kaa = KAA(k=k,
                  data=edge_list,
                  data_type='edge list',
                  type='jaccard',
                  link_pred=True,
                  seed_split = seed_split,
                  seed_init = seed_init
                  )
        kaa.train(iterations=iter)
        kaa_models.append(kaa)

        kaainit = KAA(k=k,
                  data=edge_list,
                  data_type="edge list",
                  link_pred=True,
                  seed_split=seed_split,
                  seed_init=seed_init
                  )
        kaainit.train(iterations=1000)

        raainit = DRRAA(init_Z=kaainit.S.detach(),
                        k=k,
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
        #############################################################################
        #NMIs - require full data, so link_pred=False, else everything is the same :)
        raa_nmi = DRRAA(k=k,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type="edge list",
                    link_pred=False,
                    seed_init=seed_init
                    )
        raa_nmi.train(iterations=iter)
        raa_nmi_models.append(raa_nmi)

        lsm_nmi = LSM(d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type="edge list",
                    link_pred=False,
                    seed_init=seed_init
                    )
        lsm_nmi.train(iterations=iter)
        lsm_nmi_models.append(lsm_nmi)

        lsmaa_nmi = LSMAA(d=d,
                      k=k,
                      sample_size=1,
                      data = edge_list,
                      data_type = "Edge list",
                      link_pred = False,
                      seed_split=seed_split
                      )
        lsmaa_nmi.train(iterations=iter)
        lsmaa_nmi_models.append(lsmaa_nmi)

        kaa_nmi = KAA(k=k,
                  data=adj_m.numpy(),
                  type='jaccard',
                  link_pred=False,
                  seed_split = seed_split
                  )
        kaa_nmi.train(iterations=iter)
        kaa_nmi_models.append(kaa_nmi)

        kaainit = KAA(k=k,
                      data=adj_m.numpy(),
                      link_pred=False,
                      seed_init=seed_init
                      )
        kaainit.train(iterations=1000)

        raainit_nmi = DRRAA(init_Z=kaainit.S.detach(),
                        k=k,
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

    raa_aucs = []
    lsm_aucs = []
    lsmaa_aucs = []
    kaa_aucs = []
    raainit_aucs = []

    raa_nmis = []
    lsm_nmis = []
    lsmaa_nmis = []
    kaa_nmis = []
    raainit_nmis = []
    for i in range(num_init):
        #calc aucs
        raa_auc, _, _ = raa_models[i].link_prediction()
        lsm_auc, _, _ = lsm_models[i].link_prediction()
        lsmaa_auc, _, _ = lsmaa_models[i].link_prediction()
        kaa_auc, _, _ = kaa_models[i].link_prediction()
        raainit_auc, _, _ = raainit_models[i].link_prediction()

        raa_aucs.append(raa_auc)
        lsm_aucs.append(lsm_auc)
        lsmaa_aucs.append(lsmaa_auc)
        kaa_aucs.append(kaa_auc)
        raainit_aucs.append(raainit_auc)

        #calc nmis

        raa_nmi = calcNMI(F.softmax(raa_nmi_models[i].Z.detach(),dim=0), Z_true)
        lsm_nmi = calcNMI(F.normalize(lsm_nmi_models[i].latent_Z.detach().T,dim=0), Z_true) #TODO: normalise instead of softmax
        aa = arch.AA(n_archetypes=k)
        Z = aa.fit_transform(lsmaa_nmi_models[i].latent_Z.detach().numpy())
        lsmaa_nmi = calcNMI(torch.from_numpy(Z).T.float(), Z_true)
        kaa_nmi = calcNMI(F.softmax(kaa_nmi_models[i].S.detach(),dim=0), Z_true)
        raainit_nmi = calcNMI(F.softmax(raainit_nmi_models[i].Z.detach(),dim=0), Z_true)

        raa_nmis.append(raa_nmi)
        lsm_nmis.append(lsm_nmi)
        lsmaa_nmis.append(lsmaa_nmi)
        kaa_nmis.append(kaa_nmi)
        raainit_nmis.append(raainit_nmi)


    #append aucs and NMIs
    raa_best_in_seed_aucs[alpha_idx,:] = raa_aucs
    lsm_best_in_seed_aucs[alpha_idx,:] = lsm_aucs
    lsmaa_best_in_seed_aucs[alpha_idx,:] = lsmaa_aucs
    kaa_best_in_seed_aucs[alpha_idx,:] = kaa_aucs
    raainit_best_in_seed_aucs[alpha_idx, :] = raainit_aucs

    raa_best_in_seed_nmis[alpha_idx,:] = raa_nmis
    lsm_best_in_seed_nmis[alpha_idx,:] = lsm_nmis
    lsmaa_best_in_seed_nmis[alpha_idx,:] = lsmaa_nmis
    kaa_best_in_seed_nmis[alpha_idx,:] = kaa_nmis
    raainit_best_in_seed_nmis[alpha_idx, :] = raainit_nmis

avg_raa_aucs = np.mean(raa_best_in_seed_aucs,1)
avg_lsm_aucs = np.mean(lsm_best_in_seed_aucs,1)
avg_lsmaa_aucs = np.mean(lsmaa_best_in_seed_aucs,1)
avg_kaa_aucs = np.mean(kaa_best_in_seed_aucs,1)
avg_raainit_aucs = np.mean(raainit_best_in_seed_aucs,1)

conf_raa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_aucs,
                        scale=st.sem(raa_best_in_seed_aucs,1))
conf_lsm_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_lsm_aucs,
                        scale=st.sem(lsm_best_in_seed_aucs,1))
conf_lsmaa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_lsmaa_aucs,
                        scale=st.sem(lsmaa_best_in_seed_aucs,1))
conf_kaa_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_aucs,
                        scale=st.sem(kaa_best_in_seed_aucs,1))
conf_raainit_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raainit_aucs,
                        scale=st.sem(raainit_best_in_seed_aucs,1))

avg_raa_nmis = np.mean(raa_best_in_seed_nmis,1)
avg_lsm_nmis = np.mean(lsm_best_in_seed_nmis,1)
avg_lsmaa_nmis = np.mean(lsmaa_best_in_seed_nmis,1)
avg_kaa_nmis = np.mean(kaa_best_in_seed_nmis,1)
avg_raainit_nmis = np.mean(raainit_best_in_seed_nmis,1)

conf_raa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raa_nmis,
                        scale=st.sem(raa_best_in_seed_nmis,1))
conf_lsm_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_lsm_nmis,
                        scale=st.sem(lsm_best_in_seed_nmis,1))
conf_lsmaa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_lsmaa_nmis,
                        scale=st.sem(lsmaa_best_in_seed_nmis,1))
conf_kaa_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_nmis,
                        scale=st.sem(kaa_best_in_seed_nmis,1))
conf_raainit_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_raainit_nmis,
                        scale=st.sem(raainit_best_in_seed_nmis,1))

#AUC plot
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(alphas, avg_raa_aucs, '-o', label="RAA", color='#e3427d')
ax.fill_between(alphas,
                 y1 = conf_raa_aucs[0],
                 y2 = conf_raa_aucs[1],
                 color='#e3427d', alpha=0.2)
ax.plot(alphas, conf_raa_aucs[0], '--', color='#e3427d')
ax.plot(alphas, conf_raa_aucs[1], '--', color='#e3427d')

ax.plot(alphas, avg_lsm_aucs, '-o', label="LDM", color='#e68653')
ax.fill_between(alphas,
                 y1 = conf_lsm_aucs[0],
                 y2 = conf_lsm_aucs[1],
                 color='#e68653', alpha=0.2)
ax.plot(alphas, conf_lsm_aucs[0], '--', color='#e68653')
ax.plot(alphas, conf_lsm_aucs[1], '--', color='#e68653')

ax.plot(alphas, avg_lsmaa_aucs, '-o', label="LDM+AA", color='#bdb2ff')
ax.fill_between(alphas,
                 y1 = conf_lsmaa_aucs[0],
                 y2 = conf_lsmaa_aucs[1],
                 color='#bdb2ff', alpha=0.2)
ax.plot(alphas, conf_lsmaa_aucs[0], '--', color='#bdb2ff')
ax.plot(alphas, conf_lsmaa_aucs[1], '--', color='#bdb2ff')

ax.plot(alphas, avg_kaa_aucs, '-o', label="KAA", color='#ffd6a5')
ax.fill_between(alphas,
                 y1 = conf_kaa_aucs[0],
                 y2 = conf_kaa_aucs[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(alphas, conf_kaa_aucs[0], '--', color='#ffd6a5')
ax.plot(alphas, conf_kaa_aucs[1], '--', color='#ffd6a5')

ax.plot(alphas, avg_raainit_aucs, '-o', label="RAA with init", color='#746ab0')
ax.fill_between(alphas,
                 y1 = conf_raainit_aucs[0],
                 y2 = conf_raainit_aucs[1],
                 color='#746ab0', alpha=0.2)
ax.plot(alphas, conf_raainit_aucs[0], '--', color='#746ab0')
ax.plot(alphas, conf_raainit_aucs[1], '--', color='#746ab0')

ax.grid(alpha=.3)
ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("AUC")
ax.legend()
plt.savefig('two_step_alphas_auc.png',dpi=500)
#plt.show()


#NMI plot:
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(alphas, avg_raa_nmis, '-o', label="RAA", color='#e3427d')
ax.fill_between(alphas,
                 y1 = conf_raa_nmis[0],
                 y2 = conf_raa_nmis[1],
                 color='#e3427d', alpha=0.2)
ax.plot(alphas, conf_raa_nmis[0], '--', color='#e3427d')
ax.plot(alphas, conf_raa_nmis[1], '--', color='#e3427d')

ax.plot(alphas, avg_lsm_nmis, '-o', label="LDM", color='#e68653')
ax.fill_between(alphas,
                 y1 = conf_lsm_nmis[0],
                 y2 = conf_lsm_nmis[1],
                 color='#e68653', alpha=0.2)
ax.plot(alphas, conf_lsm_nmis[0], '--', color='#e68653')
ax.plot(alphas, conf_lsm_nmis[1], '--', color='#e68653')

ax.plot(alphas, avg_lsmaa_nmis, '-o', label="LDM+AA", color='#bdb2ff')
ax.fill_between(alphas,
                 y1 = conf_lsmaa_nmis[0],
                 y2 = conf_lsmaa_nmis[1],
                 color='#bdb2ff', alpha=0.2)
ax.plot(alphas, conf_lsmaa_nmis[0], '--', color='#bdb2ff')
ax.plot(alphas, conf_lsmaa_nmis[1], '--', color='#bdb2ff')

ax.plot(alphas, avg_kaa_nmis, '-o', label="KAA", color='#ffd6a5')
ax.fill_between(alphas,
                 y1 = conf_kaa_nmis[0],
                 y2 = conf_kaa_nmis[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(alphas, conf_kaa_nmis[0], '--', color='#ffd6a5')
ax.plot(alphas, conf_kaa_nmis[1], '--', color='#ffd6a5')

ax.plot(alphas, avg_raainit_nmis, '-o', label="RAA with init", color='#746ab0')
ax.fill_between(alphas,
                 y1 = conf_raainit_nmis[0],
                 y2 = conf_raainit_nmis[1],
                 color='#746ab0', alpha=0.2)
ax.plot(alphas, conf_raainit_nmis[0], '--', color='#746ab0')
ax.plot(alphas, conf_raainit_nmis[1], '--', color='#746ab0')

ax.grid(alpha=.3)
ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("NMI")
ax.legend()
plt.savefig('two_step_alphas_nmi.png',dpi=500)