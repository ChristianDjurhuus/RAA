'''
Random effects and gating (AUC): 
    Ideal predictor: True 
    Run with synthetic data K = 3 
    Alphas - 0.2, 1, 5. 
    N = 100 
    K = 3 
    D = 2 
    CV = 5 
    seed = 42
    Sample_size = 1 
    Lr = 0.01 (default)
    Iterations = 10,000 
'''

from src.data.synthetic_data import main
from src.models.calcNMI import calcNMI

from src.models.train_DRRAA_module import DRRAA
from src.models.train_DRRAA_ngating import DRRAA_ngating
from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_DRRAA_bare import DRRAA_bare

from src.models.train_LSM_module import LSM
from src.models.train_LSM_module import LSMAA

from src.models.train_KAA_module import KAA

import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st

##########################
## Synthetic data study ##
##########################


##############################################
# Investigation of Random effects and Gating #
#       Data without Random effects          #
##############################################
#TODO
# Increasing # archetypes
# Data with and without random effects (Degree heterogenity vs not)

avgAUCs = {}
conf_AUCs = {}

avgAUCs_nr = {}
conf_AUCs_nr = {}

avgAUCs_ng = {}
conf_AUCs_ng = {}

avgAUCs_bare = {}
conf_AUCs_bare = {}

num_init = 5
k=3
d=2
nsamples=100
alphas = [0.2, 1, 5]
iter = 100
seed_split = 42
seed_init = 0
for alpha in alphas:
    #Creating synth data
    seed = 42
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    adj_m, z, A, Z_true, beta, cmap_partition = main(alpha, k, d, nsamples, rand=False)
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    temp = [x for x in nx.generate_edgelist(Graph, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)):
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    AUCs = []
    AUCs_ng = []
    AUCs_nr = []
    AUCs_bare = []

    for i in range(num_init):
        RAA = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init)

        RAA_ng = DRRAA_ngating(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init)
        
        RAA_nr = DRRAA_nre(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init)
        
        RAA_bare = DRRAA_bare(k=k,
            d=d, 
            sample_size=1, #Without random sampling
            data=edge_list,
            data_type='edge list',
            link_pred=True,
            seed_split =seed_split,
            seed_init=seed_init)

        RAA.train(iterations=iter, LR=0.1)
        RAA_ng.train(iterations=iter, LR=0.1)
        RAA_nr.train(iterations=iter, LR=0.1)
        RAA_bare.train(iterations=iter, LR=0.1)

        auc_score, fpr, tpr = RAA.link_prediction()
        auc_score_ng, fpr_ng, tpr_ng = RAA_ng.link_prediction()
        auc_score_nr, fpr_nr, tpr_nr = RAA_nr.link_prediction()
        auc_score_bare, fpr_bare, tpr_bare = RAA_bare.link_prediction()

        AUCs.append(auc_score)
        AUCs_ng.append(auc_score_ng)
        AUCs_nr.append(auc_score_nr)
        AUCs_bare.append(auc_score_bare)

    #RAA
    avgAUCs[alpha] = np.mean(AUCs)
    conf_AUCs[alpha] = st.t.interval(alpha=0.95, df=len(AUCs)-1, 
                    loc=np.mean(AUCs), 
                    scale=st.sem(AUCs))

    #RAA no gating
    avgAUCs_ng[alpha] = np.mean(AUCs_ng)
    conf_AUCs_ng[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_ng)-1, 
                    loc=np.mean(AUCs_ng), 
                    scale=st.sem(AUCs_ng))
    #RAA no random effects
    avgAUCs_nr[alpha] = np.mean(AUCs_nr)
    conf_AUCs_nr[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_nr)-1, 
                    loc=np.mean(AUCs_nr), 
                    scale=st.sem(AUCs_nr))
    
    #RAA no gate and random effects
    avgAUCs_bare[alpha] = np.mean(AUCs_bare)
    conf_AUCs_bare[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_bare)-1, 
                    loc=np.mean(AUCs_bare), 
                    scale=st.sem(AUCs_bare))

fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(alphas, list(avgAUCs.values()), '-o', label="RAA", color='#e3427d')
ax.plot(alphas, [x for (x,y) in conf_AUCs.values()], '--', color='#e3427d')
ax.plot(alphas, [y for (x,y) in conf_AUCs.values()], '--', color='#e3427d')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs.values()],
                 y2 = [y for (x,y) in conf_AUCs.values()], color='#e3427d', alpha=0.2)

ax.plot(alphas, list(avgAUCs_ng.values()), '-o', label="RAA no gating", color='#4c6e81')
ax.plot(alphas, [x for (x,y) in conf_AUCs_ng.values()], '--', color='#4c6e81')
ax.plot(alphas, [y for (x,y) in conf_AUCs_ng.values()], '--', color='#4c6e81')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_ng.values()],
                 y2 = [y for (x,y) in conf_AUCs_ng.values()], color='#4c6e81', alpha=0.2)

ax.plot(alphas, list(avgAUCs_nr.values()), '-o', label="RAA no random effects", color='#5d4b20')
ax.plot(alphas, [x for (x,y) in conf_AUCs_nr.values()], '--', color='#5d4b20')
ax.plot(alphas, [y for (x,y) in conf_AUCs_nr.values()], '--', color='#5d4b20')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_nr.values()],
                 y2 = [y for (x,y) in conf_AUCs_nr.values()], color='#5d4b20', alpha=0.2)

ax.plot(alphas, list(avgAUCs_bare.values()), '-o', label="RAA no gating / random effects", color='#f0c808')
ax.plot(alphas, [x for (x,y) in conf_AUCs_bare.values()], '--', color='#f0c808')
ax.plot(alphas, [y for (x,y) in conf_AUCs_bare.values()], '--', color='#f0c808')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_bare.values()],
                 y2 = [y for (x,y) in conf_AUCs_bare.values()], color='#f0c808', alpha=0.2)

ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("AUC")
ax.legend()
ax.grid(alpha=.3)
plt.savefig("synthetic_test.png", dpi=500)
#plt.show()


##############################################
# Investigation of Random effects and Gating #
#       Data with Random effects             #
##############################################
avgAUCs = {}
conf_AUCs = {}

avgAUCs_nr = {}
conf_AUCs_nr = {}

avgAUCs_ng = {}
conf_AUCs_ng = {}

avgAUCs_bare = {}
conf_AUCs_bare = {}

seed_init = 0
for alpha in alphas:
    #Creating synth data
    seed = 1
    torch.random.manual_seed(seed)
    np.random.seed(seed)   
    adj_m, z, A, Z_true, beta, cmap_partition = main(alpha, k, d, nsamples, rand=True)
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    temp = [x for x in nx.generate_edgelist(Graph, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)):
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    AUCs = []
    AUCs_ng = []
    AUCs_nr = []
    AUCs_bare = []

    for i in range(num_init):
        RAA = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init)

        RAA_ng = DRRAA_ngating(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init)
        
        RAA_nr = DRRAA_nre(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    seed_split = seed_split,
                    seed_init=seed_init)
        
        RAA_bare = DRRAA_bare(k=k,
            d=d, 
            sample_size=1, #Without random sampling
            data=edge_list,
            data_type='edge list',
            link_pred=True,
            seed_split = seed_split,
            seed_init=seed_init)

        RAA.train(iterations=iter, LR=0.1)
        RAA_ng.train(iterations=iter, LR=0.1)
        RAA_nr.train(iterations=iter, LR=0.1)
        RAA_bare.train(iterations=iter, LR=0.1)

        auc_score, fpr, tpr = RAA.link_prediction()
        auc_score_ng, fpr_ng, tpr_ng = RAA_ng.link_prediction()
        auc_score_nr, fpr_nr, tpr_nr = RAA_nr.link_prediction()
        auc_score_bare, fpr_bare, tpr_bare = RAA_bare.link_prediction()

        AUCs.append(auc_score)
        AUCs_ng.append(auc_score_ng)
        AUCs_nr.append(auc_score_nr)
        AUCs_bare.append(auc_score_bare)
        seed_init += 1
    #RAA
    avgAUCs[alpha] = np.mean(AUCs)
    conf_AUCs[alpha] = st.t.interval(alpha=0.95, df=len(AUCs)-1, 
                    loc=np.mean(AUCs), 
                    scale=st.sem(AUCs))

    #RAA no gating
    avgAUCs_ng[alpha] = np.mean(AUCs_ng)
    conf_AUCs_ng[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_ng)-1, 
                    loc=np.mean(AUCs_ng), 
                    scale=st.sem(AUCs_ng))
    #RAA no random effects
    avgAUCs_nr[alpha] = np.mean(AUCs_nr)
    conf_AUCs_nr[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_nr)-1, 
                    loc=np.mean(AUCs_nr), 
                    scale=st.sem(AUCs_nr))
    
    #RAA no gate and random effects
    avgAUCs_bare[alpha] = np.mean(AUCs_bare)
    conf_AUCs_bare[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_bare)-1, 
                    loc=np.mean(AUCs_bare), 
                    scale=st.sem(AUCs_bare))

fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(alphas, list(avgAUCs.values()), '-o', label="RAA", color='#e3427d')
ax.plot(alphas, [x for (x,y) in conf_AUCs.values()], '--', color='#e3427d')
ax.plot(alphas, [y for (x,y) in conf_AUCs.values()], '--', color='#e3427d')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs.values()],
                 y2 = [y for (x,y) in conf_AUCs.values()], color='#e3427d', alpha=0.2)

ax.plot(alphas, list(avgAUCs_ng.values()), '-o', label="RAA no gating", color='#4c6e81')
ax.plot(alphas, [x for (x,y) in conf_AUCs_ng.values()], '--', color='#4c6e81')
ax.plot(alphas, [y for (x,y) in conf_AUCs_ng.values()], '--', color='#4c6e81')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_ng.values()],
                 y2 = [y for (x,y) in conf_AUCs_ng.values()], color='#4c6e81', alpha=0.2)

ax.plot(alphas, list(avgAUCs_nr.values()), '-o', label="RAA no random effects", color='#5d4b20')
ax.plot(alphas, [x for (x,y) in conf_AUCs_nr.values()], '--', color='#5d4b20')
ax.plot(alphas, [y for (x,y) in conf_AUCs_nr.values()], '--', color='#5d4b20')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_nr.values()],
                 y2 = [y for (x,y) in conf_AUCs_nr.values()], color='#5d4b20', alpha=0.2)

ax.plot(alphas, list(avgAUCs_bare.values()), '-o', label="RAA no gating / random effects", color='#f0c808')
ax.plot(alphas, [x for (x,y) in conf_AUCs_bare.values()], '--', color='#f0c808')
ax.plot(alphas, [y for (x,y) in conf_AUCs_bare.values()], '--', color='#f0c808')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_bare.values()],
                 y2 = [y for (x,y) in conf_AUCs_bare.values()], color='#f0c808', alpha=0.2)

ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("AUC")
ax.legend()
ax.grid(alpha=.3)
plt.savefig("synthetic_test_re.png", dpi=500)
#plt.show()