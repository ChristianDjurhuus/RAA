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

rand=True

#set test and train split seed. We want the same train and test split in order to know that differences
#are because of inits.
seed_split = 42

#Run 10 different seeds with 10 different inits. Take the best of the 10 inits and save as best in seed.
#Then plot the auc and nmi with errorbars on the 10 best in seeds.
alphas = [0.2, 1, 5]
num_init = 10

kaa_na_best_in_seed_aucs = np.zeros((len(alphas),num_init))
kaa_ja_best_in_seed_aucs = np.zeros((len(alphas),num_init))

kaa_na_best_in_seed_nmis = np.zeros((len(alphas),num_init))
kaa_ja_best_in_seed_nmis = np.zeros((len(alphas),num_init))

seed_init = 100


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

    k= 3
    n = 100
    d = 2
    adj_m, z, A, Z_true, beta, partition = main(alpha=alpha, k=k, dim=d, nsamples=n, rand=rand)
    G = nx.from_numpy_matrix(adj_m.numpy())

    temp = [x for x in nx.generate_edgelist(G, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)):
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    kaa_na_models = []
    kaa_ja_models = []

    kaa_na_nmi_models = []
    kaa_ja_nmi_models = []

    for init in range(num_init):
        kaa_na = KAA(k=k,
                  data=edge_list,
                  data_type='edge list',
                  type='normalised adjacency',
                  link_pred=True,
                  seed_split = seed_split,
                  seed_init = seed_init
                  )
        kaa_na.train(iterations=iter)
        kaa_na_models.append(kaa_na)

        kaa_ja = KAA(k=k,
                  data=edge_list,
                  data_type="edge list",
                  type = 'jaccard',
                  link_pred=True,
                  seed_split=seed_split,
                  seed_init=seed_init
                  )
        kaa_ja.train(iterations=iter)
        kaa_ja_models.append(kaa_ja)
        #############################################################################
        #NMIs - require full data, so link_pred=False, else everything is the same :)
        kaa_na_nmi = KAA(k=k,
                  data=adj_m.numpy(),
                  type='normalised adjacency',
                  link_pred=False,
                  seed_split = seed_split
                  )
        kaa_na_nmi.train(iterations=iter)
        kaa_na_nmi_models.append(kaa_na_nmi)
        
        kaa_ja_nmi = KAA(k=k,
                      data=adj_m.numpy(),
                      link_pred=False,
                      type = 'jaccard',
                      seed_init=seed_init
                      )
        kaa_ja_nmi.train(iterations=iter)
        kaa_ja_nmi_models.append(kaa_ja_nmi)

        #make sure to increase the initialisation-seed ;)
        seed_init += 1
        print(seed_init)


    kaa_na_aucs = []
    kaa_ja_aucs = []


    kaa_na_nmis = []
    kaa_ja_nmis = []
    for i in range(num_init):
        #calc aucs
        kaa_na_auc, _, _ = kaa_na_models[i].link_prediction()
        kaa_ja_auc, _, _ = kaa_ja_models[i].link_prediction()

        kaa_na_aucs.append(kaa_na_auc)
        kaa_ja_aucs.append(kaa_ja_auc)

        #calc nmis

        kaa_na_nmi = calcNMI(F.softmax(kaa_na_nmi_models[i].S.detach(),dim=0), Z_true)
        kaa_ja_nmi = calcNMI(F.softmax(kaa_ja_nmi_models[i].S.detach(),dim=0), Z_true)

        kaa_na_nmis.append(kaa_na_nmi)
        kaa_ja_nmis.append(kaa_ja_nmi)


    #append aucs and NMIs
    kaa_na_best_in_seed_aucs[alpha_idx,:] = kaa_na_aucs
    kaa_ja_best_in_seed_aucs[alpha_idx, :] = kaa_ja_aucs

    kaa_na_best_in_seed_nmis[alpha_idx,:] = kaa_na_nmis
    kaa_ja_best_in_seed_nmis[alpha_idx, :] = kaa_ja_nmis

avg_kaa_na_aucs = np.mean(kaa_na_best_in_seed_aucs,1)
avg_kaa_ja_aucs = np.mean(kaa_ja_best_in_seed_aucs,1)

conf_kaa_na_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_na_aucs,
                        scale=st.sem(kaa_na_best_in_seed_aucs,1))
conf_kaa_ja_aucs = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_ja_aucs,
                        scale=st.sem(kaa_ja_best_in_seed_aucs,1))

avg_kaa_na_nmis = np.mean(kaa_na_best_in_seed_nmis,1)
avg_kaa_ja_nmis = np.mean(kaa_ja_best_in_seed_nmis,1)

conf_kaa_na_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_na_nmis,
                        scale=st.sem(kaa_na_best_in_seed_nmis,1))
conf_kaa_ja_nmis = st.t.interval(alpha=0.95, df=num_init-1,
                        loc=avg_kaa_ja_nmis,
                        scale=st.sem(kaa_ja_best_in_seed_nmis,1))

#AUC plot
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(alphas, avg_kaa_na_aucs, '-o', label="KAA with normalised adjacency kernel", color='#746ab0')
ax.fill_between(alphas,
                 y1 = conf_kaa_na_aucs[0],
                 y2 = conf_kaa_na_aucs[1],
                 color='#746ab0', alpha=0.2)
ax.plot(alphas, conf_kaa_na_aucs[0], '--', color='#746ab0')
ax.plot(alphas, conf_kaa_na_aucs[1], '--', color='#746ab0')

ax.plot(alphas, avg_kaa_ja_aucs, '-o', label="KAA with jaccard kernel", color='#ffd6a5')
ax.fill_between(alphas,
                 y1 = conf_kaa_ja_aucs[0],
                 y2 = conf_kaa_ja_aucs[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(alphas, conf_kaa_ja_aucs[0], '--', color='#ffd6a5')
ax.plot(alphas, conf_kaa_ja_aucs[1], '--', color='#ffd6a5')

ax.grid(alpha=.3)
ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("AUC")
ax.legend()
plt.savefig('normalised_adjacency_alphas_auc_rand2.png',dpi=500)
#plt.show()


#NMI plot:
fig, ax = plt.subplots(figsize=(7,5), dpi=500)
ax.plot(alphas, avg_kaa_na_nmis, '-o', label="KAA with normalised adjacency kernel", color='#746ab0')
ax.fill_between(alphas,
                 y1 = conf_kaa_na_nmis[0],
                 y2 = conf_kaa_na_nmis[1],
                 color='#746ab0', alpha=0.2)
ax.plot(alphas, conf_kaa_na_nmis[0], '--', color='#746ab0')
ax.plot(alphas, conf_kaa_na_nmis[1], '--', color='#746ab0')

ax.plot(alphas, avg_kaa_ja_nmis, '-o', label="KAA with jaccard kernel", color='#ffd6a5')
ax.fill_between(alphas,
                 y1 = conf_kaa_ja_nmis[0],
                 y2 = conf_kaa_ja_nmis[1],
                 color='#ffd6a5', alpha=0.2)
ax.plot(alphas, conf_kaa_ja_nmis[0], '--', color='#ffd6a5')
ax.plot(alphas, conf_kaa_ja_nmis[1], '--', color='#ffd6a5')

ax.grid(alpha=.3)
ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("NMI")
ax.legend()
plt.savefig('normalised_adjacency_alphas_nmi_rand2.png',dpi=500)