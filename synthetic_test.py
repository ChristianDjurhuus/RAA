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
##############################################
#TODO
# Increasing # archetypes
# Data with and without random effects (Degree heterogenity vs not)
seed = 1998
torch.random.manual_seed(seed)
np.random.seed(seed)

avgAUCs = {}
avgNMIs = {}
conf_NMIs = {}
conf_AUCs = {}

avgAUCs_nr = {}
avgNMIs_nr = {}
conf_NMIs_nr = {}
conf_AUCs_nr = {}

avgAUCs_ng = {}
avgNMIs_ng = {}
conf_NMIs_ng = {}
conf_AUCs_ng = {}

avgAUCs_bare = {}
avgNMIs_bare = {}
conf_NMIs_bare = {}
conf_AUCs_bare = {}


num_init = 5
k=3
d=2
nsamples=100
alphas = [0.2, 1, 5]
iter = 10000
for alpha in alphas:
    #Creating synth data
    adj_m, z, A, Z_true = main(alpha, k, d, nsamples)
    Graph = nx.from_numpy_matrix(adj_m.numpy())

    AUCs = []
    AUCs_ng = []
    AUCs_nr = []
    AUCs_bare = []
    NMIs = []
    NMIs_ng = []
    NMIs_nr = []
    NMIs_bare = []

    for i in range(num_init):
        RAA = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=Graph,
                    data_type='networkx',
                    link_pred=True)
        RAA_ng = DRRAA_ngating(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=Graph,
                    data_type='networkx',
                    link_pred=True)
        
        RAA_nr = DRRAA_nre(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=Graph,
                    data_type='networkx',
                    link_pred=True)
        
        RAA_bare = DRRAA_bare(k=k,
            d=d, 
            sample_size=1, #Without random sampling
            data=Graph,
            data_type='networkx',
            link_pred=True)

        RAA.train(iterations=iter, LR=0.01)
        RAA_ng.train(iterations=iter, LR=0.01)
        RAA_nr.train(iterations=iter, LR=0.01)
        RAA_bare.train(iterations=iter, LR=0.01)

        auc_score, fpr, tpr = RAA.link_prediction()
        auc_score_ng, fpr_ng, tpr_ng = RAA_ng.link_prediction()
        auc_score_nr, fpr_nr, tpr_nr = RAA_nr.link_prediction()
        auc_score_bare, fpr_bare, tpr_bare = RAA_bare.link_prediction()

        AUCs.append(auc_score)
        AUCs_ng.append(auc_score_ng)
        AUCs_nr.append(auc_score_nr)
        AUCs_bare.append(auc_score_bare)

        #RAA
        Z = F.softmax(RAA.Z, dim=0)
        #RAA no gating
        Z_ng = F.softmax(RAA_ng.Z, dim=0)
        #RAA no random effects
        Z_nr = F.softmax(RAA_nr.Z, dim=0)
        #RAA no gate no RE
        Z_bare = F.softmax(RAA_bare.Z, dim=0)

        NMIs.append(calcNMI(Z, Z_true).item())
        NMIs_ng.append(calcNMI(Z_ng, Z_true).item())
        NMIs_nr.append(calcNMI(Z_nr, Z_true).item())
        NMIs_bare.append(calcNMI(Z_bare, Z_true).item())

    #RAA
    avgNMIs[alpha] = np.mean(NMIs)
    conf_NMIs[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs)-1, 
                    loc=np.mean(NMIs), 
                    scale=st.sem(NMIs)) 
    avgAUCs[alpha] = np.mean(AUCs)
    conf_AUCs[alpha] = st.t.interval(alpha=0.95, df=len(AUCs)-1, 
                    loc=np.mean(AUCs), 
                    scale=st.sem(AUCs))

    #RAA no gating
    avgNMIs_ng[alpha] = np.mean(NMIs_ng)
    conf_NMIs_ng[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_ng)-1, 
                    loc=np.mean(NMIs_ng), 
                    scale=st.sem(NMIs_ng)) 
    avgAUCs_ng[alpha] = np.mean(AUCs_ng)
    conf_AUCs_ng[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_ng)-1, 
                    loc=np.mean(AUCs_ng), 
                    scale=st.sem(AUCs_ng))
    #RAA no random effects
    avgNMIs_nr[alpha] = np.mean(NMIs_nr)
    conf_NMIs_nr[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_nr)-1, 
                    loc=np.mean(NMIs_nr), 
                    scale=st.sem(NMIs_nr)) 
    avgAUCs_nr[alpha] = np.mean(AUCs_nr)
    conf_AUCs_nr[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_nr)-1, 
                    loc=np.mean(AUCs_nr), 
                    scale=st.sem(AUCs_nr))
    
    #RAA no gate and random effects
    avgNMIs_bare[alpha] = np.mean(NMIs_bare)
    conf_NMIs_bare[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_bare)-1, 
                    loc=np.mean(NMIs_bare), 
                    scale=st.sem(NMIs_bare)) 
    avgAUCs_bare[alpha] = np.mean(AUCs_bare)
    conf_AUCs_bare[alpha] = st.t.interval(alpha=0.95, df=len(AUCs_bare)-1, 
                    loc=np.mean(AUCs_bare), 
                    scale=st.sem(AUCs_bare))

mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(alphas, list(avgNMIs.values()), '-o', label="RAA")
ax.plot(alphas, [x for (x,y) in conf_NMIs.values()], '--', color='C0')
ax.plot(alphas, [y for (x,y) in conf_NMIs.values()], '--', color='C0')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs.values()],
                 y2 = [y for (x,y) in conf_NMIs.values()], color='C0', alpha=0.2)

ax.plot(alphas, list(avgNMIs_ng.values()), '-o', label="RAA no gating")
ax.plot(alphas, [x for (x,y) in conf_NMIs_ng.values()], '--', color='C1')
ax.plot(alphas, [y for (x,y) in conf_NMIs_ng.values()], '--', color='C1')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_ng.values()],
                 y2 = [y for (x,y) in conf_NMIs_ng.values()], color='C1', alpha=0.2)

ax.plot(alphas, list(avgNMIs_nr.values()), '-o', label="RAA no random effects")
ax.plot(alphas, [x for (x,y) in conf_NMIs_nr.values()], '--', color='C2')
ax.plot(alphas, [y for (x,y) in conf_NMIs_nr.values()], '--', color='C2')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_nr.values()],
                 y2 = [y for (x,y) in conf_NMIs_nr.values()], color='C2', alpha=0.2)

ax.plot(alphas, list(avgNMIs_bare.values()), '-o', label="RAA no gating / random effects")
ax.plot(alphas, [x for (x,y) in conf_NMIs_bare.values()], '--', color='C3')
ax.plot(alphas, [y for (x,y) in conf_NMIs_bare.values()], '--', color='C3')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_bare.values()],
                 y2 = [y for (x,y) in conf_NMIs_bare.values()], color='C3', alpha=0.2)                 

ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("NMI")
ax.grid(alpha=.3)
ax.legend()
#plt.savefig("RAA_properties_NMI.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(alphas, list(avgAUCs.values()), '-o', label="RAA")
ax.plot(alphas, [x for (x,y) in conf_AUCs.values()], '--', color='C0')
ax.plot(alphas, [y for (x,y) in conf_AUCs.values()], '--', color='C0')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs.values()],
                 y2 = [y for (x,y) in conf_AUCs.values()], color='C0', alpha=0.2)

ax.plot(alphas, list(avgAUCs_ng.values()), '-o', label="RAA no gating")
ax.plot(alphas, [x for (x,y) in conf_AUCs_ng.values()], '--', color='C1')
ax.plot(alphas, [y for (x,y) in conf_AUCs_ng.values()], '--', color='C1')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_ng.values()],
                 y2 = [y for (x,y) in conf_AUCs_ng.values()], color='C1', alpha=0.2)

ax.plot(alphas, list(avgAUCs_nr.values()), '-o', label="RAA no random effects")
ax.plot(alphas, [x for (x,y) in conf_AUCs_nr.values()], '--', color='C2')
ax.plot(alphas, [y for (x,y) in conf_AUCs_nr.values()], '--', color='C2')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_nr.values()],
                 y2 = [y for (x,y) in conf_AUCs_nr.values()], color='C2', alpha=0.2)

ax.plot(alphas, list(avgAUCs_bare.values()), '-o', label="RAA no gating / random effects")
ax.plot(alphas, [x for (x,y) in conf_AUCs_bare.values()], '--', color='C3')
ax.plot(alphas, [y for (x,y) in conf_AUCs_bare.values()], '--', color='C3')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_AUCs_bare.values()],
                 y2 = [y for (x,y) in conf_AUCs_bare.values()], color='C3', alpha=0.2)

ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("AUC")
ax.legend()
ax.grid(alpha=.3)
#plt.savefig("RAA_properties_AUC.pdf")
plt.show()


###########################
####    Comparison      ###
#### RAA and baselines  ###
###########################






